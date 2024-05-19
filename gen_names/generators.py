from typing import Literal, Any, List
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
    def __init__(
            self, 
            model, 
            tokenizer, 
            device:Literal["cuda", "cpu"] = 'cuda', 
            eos_token_id:int = 3,
            min_lenght:int = 4,
            pad_value:int = 0
        ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.model.to(self.device)
        self.eos_token_id = eos_token_id
        self.chunk_lenght = min_lenght
        self.pad_value = pad_value
    
    def ensure_length(self, txt:str) -> str:
        if len(txt) < self.chunk_lenght:
            txt = list(txt) + [self.pad_value] * (self.chunk_lenght - len(txt))
        
        return txt
    
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
        # Если есть параметр альфа, его применяем по формуле
        if alpha != 0:
            original = (1 - alpha) * original + alpha / len(original)
        # Делим логарифм весов на температуру для усреднения весов, сила которого зависит от температуры
        distribution = original / temperature

        return distribution

    def __call__(
            self, 
            seed_text, 
            max_steps_n=40, 
            return_hypotheses_n=5, 
            beamsize=5, 
            temperature=0.5, 
            alpha=0, 
            need_reweight=False, 
            without_score=False, 
            need_to_encode=True
        ):
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
            in_batch = torch.tensor(cur_partial_hypothesis)
            # ------------------------------------------------------
            # Здесь мы добавляем паддингом до необходимой длины, чтобы conv в модели не грохнулся
            # ------------------------------------------------------
            # in_batch = self.ensure_length(in_batch)
            in_batch = torch.Tensor(in_batch).to(torch.int).to(self.device)
            in_batch = in_batch.unsqueeze(0)
            # ------------------------------------------------------
            # ------------------------------------------------------
            next_tokens_logits = self.model(in_batch)[0, -1]

            # При необходимости перевзвешиваем веса
            #-------------------------------------------------------
            # В дипломе веса даже не перевзвешивались, гений
            #-------------------------------------------------------
            if need_reweight:
                # next_tokens_logproba = self.reweight(next_tokens_logits, temperature, alpha)
                next_tokens_logits = self.reweight(next_tokens_logits, temperature, alpha)
            
            # Выбираем топ-beamsize лучших вариантов токенов по весам
            next_tokens_logproba = F.log_softmax(next_tokens_logits, dim=-1)
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
        final_texts = self.tokenizer.decode_many(list(final_token_lists))

        result = list(zip(final_scores, final_texts))
        result.sort()
        result = result[:return_hypotheses_n]

        if without_score:
            final_scores, result = zip(*result)
        
        return result

class TextGenerator:
    def __init__(
            self,
            model:Any,
            tokenizer:Any,
            banned_idx:List[int]|None = None,
            device:Literal["cpu", "cuda"] = "cuda",
            eos_token_id:int = 3,
        ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        if banned_idx is None:
            banned_idx = [tokenizer.pad_value]
        self.banned_idx = banned_idx.copy()
        self.banned_idx_origin = banned_idx.copy()

        self.repeat_tokens = {}

        self.device = device
        self.eos_token_id = eos_token_id

    def phrase_to_tensor(self, phrase:str) -> torch.Tensor:
        text = self.tokenizer.encode(phrase)
        text = torch.Tensor(text).to(torch.int).unsqueeze(0).to(self.device)
        return text
    
    def choose_idx(self, sa_cs: torch.Tensor) -> int:
        rd = torch.rand(1).to(self.device)
        ch = 0
        for i in range(len(sa_cs)):
            if sa_cs[i] > rd:
                ch = i
                break
        return ch
    
    def cut_sort(self, sort:torch.Tensor, N:int) -> torch.Tensor:
        sort_cut = sort[:N]
        removed_all_time = 0
        while True:
            been_removed = 0
            for i in range(len(sort_cut)-1):
                try:
                    if sort_cut[i] in self.banned_idx:
                        sort_cut = torch.cat([sort_cut[:i], sort_cut[i+1:]])
                        been_removed += 1
                except:
                    pass
            if sort_cut[-1] in self.banned_idx:
                sort_cut = sort_cut[:-1]
                been_removed += 1
            
            if been_removed > 0:
                sort_cut = torch.cat([
                    sort_cut, 
                    sort[N+removed_all_time : N+removed_all_time+been_removed]
                ])
                removed_all_time += been_removed
            
            if been_removed == 0:
                break

        return sort_cut
    
    def check_repeat(self, token:int, max_repeat:int) -> bool:
        try:
            how_many = self.repeat_tokens[token]
            if how_many < max_repeat:
                self.repeat_tokens[token] += 1
            else:
                return False
        except:
            if max_repeat > 1:
                self.repeat_tokens[token] = 1
            else:
                return False
        return True

    def gen_tk(self, phrase: torch.Tensor, N:int) -> torch.Tensor:
        answer = self.model(phrase)[0][-1]
        sort = torch.argsort(answer, descending=True)
        sort_cut = self.cut_sort(sort, N)

        sort_answer = []
        answer = answer.softmax(dim=0)
        for s in sort_cut:
            sort_answer.append(answer[s])
        sort_answer = torch.stack(sort_answer)
        sa_cumsum = (sort_answer / sort_answer.sum()).cumsum(dim=0)

        ch_num = self.choose_idx(sa_cumsum)
        idx = sort_cut[ch_num]

        return idx
    
    def gen_tn(self, phrase: torch.Tensor, k:float) -> torch.Tensor:
        answer = self.model(phrase)[0][-1]
        sort = torch.sort(answer, descending=True)

        sw_cumsum = sort.values.softmax(dim=0).cumsum(dim=0)

        for i in range(len(sw_cumsum)):
            if sw_cumsum[i] > k:
                break
        
        sort_cut = self.cut_sort(sort.indices, i+1)

        sw_cut = sort.values.softmax(dim=0)[:i]
        sa_cumsum = (sw_cut / sw_cut.sum()).cumsum(dim=0)

        ch_num = self.choose_idx(sa_cumsum)
        idx = sort_cut[ch_num]

        return idx

    def generate(
            self, 
            phrase:str = "F",
            mode:Literal["top_k", "top_n"] = "top_k",
            k:int = 5,
            n:float = 0.9,
            max_len:int = 16,
            max_repeat:int|None = None
        ) -> str:
        for _ in range(max_len):
            phrase_ts = self.phrase_to_tensor(phrase)
            
            if mode == "top_k":
                next_idx = self.gen_tk(phrase_ts, k)
            if mode == "top_n":
                next_idx = self.gen_tn(phrase_ts, n)

            if next_idx == self.eos_token_id:
                break
            phrase = phrase + self.tokenizer.decode([next_idx])

            if max_repeat is not None:
                if not self.check_repeat(next_idx.item(), max_repeat):
                    self.banned_idx.append(next_idx.item())
        
        self.banned_idx = self.banned_idx_origin.copy()
        return phrase