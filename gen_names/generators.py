import torch
import heapq
import numpy as np
import torch.nn.functional as F

class BeamGenerator:
    """
    Поисково-лучевая генерация продолжения предложения\n
    При созданиии BeamGenerator:
    :param model: - Модель для генерации
    :param tokenizer: - токенизатор для перевода текста с токенов в символы и обратно
    :param device: cuda/cpu - где будут производиться вычисления (по умолчанию cuda)
    :param eos_token_id: - номер токена, означающий конец предложения (End Of String, <EOS>)\n
    При вызове BeamGenerator:
    :param seed_text: - "Начало" предложения, по которому будет производится генерация
    :param max_steps_n: - длина сгенерированных предложений
    :param return_hypotheses_n: - сколько возвращает "гипотез" продолжения предложений (по умолчанию = 5)
    :param beamsize: - ширина луча (по умолчанию = 5)
    :param temperature: float (от 0.0 до 1.0) - температура генерации, который уменьшает "уверенность" модели в выборе следующего токена (по умолчанию = 0.5)
    :param alpha: float (от 0.0 до 1.0) - параметр для перевзвешивания для уменьшения "уверенности" модели в выборе следующего токена (по умолчанию = 0)
    :param need_reweight: - ключ, будет ли перевзвешивание весов ответов для уменьшения "уверенности" модели (по умолчанию False)
    :param without_score: - ключ, нужно ли возвращать дополнительно общий вес сгенерированного продолжения предложения (по умолчанию = False)
    :param need_to_encode: - ключ, нужно ли переводить seed_text в токены (по умолчанию = True. Изменить на False, если предложение подаётся сразу в токенах)
    :return: 
        - если without_score = False, то сгенерированное продолжение предложения
        - если without_score = True, то кортеж из двух элементов: 
            - вес предложения
            - сгенерированное продолжение предложения
        
    """
    def __init__(self, model, tokenizer, device='cuda', eos_token_id=3):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.model.to(self.device)
        self.eos_token_id = eos_token_id
    
    def reweight(self, original, temperature=0.5, alpha=0):
        """
        Функция для перевзвешивания весов по формуле с двумя параметрами:
            - температура
            - альфа\n
        Понижает "уверенность" модели в предсказанном токене для улучшения генерации
        :param original: - изначальное распределение весов
        :param temperature: - параметр температура
        :param alpha: - параметр альфа
        :return: новое распределение весов
        """
        # Перевод массива весов в numpy
        # original = original.cpu().detach().numpy()

        # Если есть параметр альфа, его применяем по формуле
        if alpha != 0:
            original = (1 - alpha) * original + alpha / len(original)
        # Делим логарифм весов на температуру для усреднения весов, сила которого зависит от температуры
        distribution = np.log(original) / temperature

        # Перевод перевзвешенного массива весов обратно в тензор
        # distribution = torch.tensor(distribution).to(self.device)
        return distribution

    def __call__(self, seed_text, max_steps_n=40, return_hypotheses_n=5, beamsize=5, temperature=0.5, alpha=0, need_reweight=False, without_score=False, need_to_encode=True):
        # При необходимости переводим предложение из символов в токены
        if need_to_encode:
            seed_tokens = self.tokenizer.encode([seed_text])
        else:
            seed_tokens = seed_text
        initial_length = len(seed_tokens)

        partial_hypotheses = [(0, seed_tokens)]
        final_hypotheses = []

        while len(partial_hypotheses) > 0:
            # Создаём очередь для весов и токенов
            cur_partial_score, cur_partial_hypothesis = heapq.heappop(partial_hypotheses)
            
            # Генерируем первый токен
            in_batch = torch.tensor(cur_partial_hypothesis).unsqueeze(0).to(self.device)
            next_tokens_logits = self.model(in_batch)[0, -1]

            # При необходимости перевзвешиваем веса
            #-------------------------------------------------------
            # В дипломе веса даже не перевзвешивались, гений
            #-------------------------------------------------------
            if need_reweight:
                # next_tokens_logproba = self.reweight(next_tokens_logits, temperature, alpha)
                next_tokens_logits = self.reweight(next_tokens_logits, temperature, alpha)
            
            # Выбираем топ-beamsize лучших вариантов токенов по весам
            next_tokens_logproba = F.log_softmax(next_tokens_logits)
            topk_continuations = next_tokens_logproba.topk(beamsize)

            for token_score, token_idx in zip(topk_continuations.values, topk_continuations.indices):
                token_score = float(token_score)
                token_idx = int(token_idx)

                # Считаем новый score для топ-beamsize вариантов
                old_denorm_score = cur_partial_score * np.sqrt(len(cur_partial_hypothesis))
                new_score = (old_denorm_score - token_score) / np.sqrt(len(cur_partial_hypothesis) + 1)

                # Берём новый токен и добавляем в конец возможного предложения
                new_hypothesis = cur_partial_hypothesis + [token_idx]
                # Создаётся новая единица - предложение с новым токеном и его вес
                new_item = (new_score, new_hypothesis)

                # Если токен конца предложения или досточная длина - записываем в финальный вариант
                if token_idx == self.eos_token_id or len(new_hypothesis) - initial_length >= max_steps_n:
                    final_hypotheses.append(new_item)
                else:
                    # Иначе добавляем в очередь
                    heapq.heappush(partial_hypotheses, new_item)

            # Если нагенерили достаточно по ширине луча поиска, лучшие (меньшие) топ-beamsize берём
            if len(partial_hypotheses) > beamsize:
                partial_hypotheses = heapq.nsmallest(beamsize, partial_hypotheses)
                heapq.heapify(partial_hypotheses)

        final_scores, final_token_lists = zip(*final_hypotheses)
        final_texts = self.tokenizer.decode(list(final_token_lists))

        result = list(zip(final_scores, final_texts))
        result.sort()
        result = result[:return_hypotheses_n]

        if without_score:
            final_scores, result = zip(*result)
        
        return result