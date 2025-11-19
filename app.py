import gradio as gr
import pandas as pd
import faiss
import numpy as np
import json
import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import defaultdict
import math
from datetime import datetime
from sentence_transformers import SentenceTransformer

# === КОНФИГУРАЦИЯ ===
EXCEL_FILE = "База знаний.xlsx"
LOGO_PATH = "logo_fox.png"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
HISTORY_FILE = "dialog_history.jsonl"

# === УЛУЧШЕННАЯ АРХИТЕКТУРА С КОНТРОЛЕМ ДЛИНЫ ===
class ResonanceAttention(nn.Module):
    def __init__(self, dim, num_waves=8):
        super().__init__()
        self.num_waves = num_waves
        self.dim = dim
        self.wave_frequencies = nn.Parameter(torch.randn(num_waves) * 0.02)
        self.wave_amplitudes = nn.Parameter(torch.ones(num_waves))
        self.wave_projection = nn.Linear(num_waves, dim)
        
    def forward(self, x):
        batch_size, seq_len, dim = x.shape
        positions = torch.arange(seq_len, device=x.device).float()
        
        wave_patterns = []
        for i in range(self.num_waves):
            freq = self.wave_frequencies[i]
            amplitude = self.wave_amplitudes[i]
            wave = amplitude * torch.sin(2 * math.pi * freq * positions / seq_len)
            wave_patterns.append(wave)
        
        wave_matrix = torch.stack(wave_patterns, dim=1)
        wave_features = self.wave_projection(wave_matrix)
        resonance = wave_features.unsqueeze(0).expand(batch_size, -1, -1)
        return x + resonance

class QuantumLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear_real = nn.Linear(input_dim, output_dim)
        self.linear_imag = nn.Linear(input_dim, output_dim)
        self.phase_shift = nn.Parameter(torch.randn(output_dim) * 0.1)
        
    def forward(self, x):
        real_part = self.linear_real(x)
        imag_part = self.linear_imag(x)
        amplitude = torch.sqrt(real_part**2 + imag_part**2 + 1e-8)
        phase = torch.atan2(imag_part, real_part) + self.phase_shift
        output = amplitude * torch.cos(phase)
        return F.gelu(output)

class WaveTransformer(nn.Module):
    def __init__(self, vocab_size, hidden_dim=256, num_layers=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, 150, hidden_dim) * 0.02)
        
        self.quantum_layers = nn.ModuleList([
            QuantumLayer(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        self.resonance_attention = nn.ModuleList([
            ResonanceAttention(hidden_dim, num_waves=6) for _ in range(num_layers)
        ])
        
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, vocab_size)
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        batch_size, seq_len = x.shape
        x = self.embedding(x)
        
        if seq_len <= 150:
            x = x + self.pos_encoding[:, :seq_len, :]
        
        x = self.dropout(x)
        
        for quantum_layer, resonance_attn in zip(self.quantum_layers, self.resonance_attention):
            x = quantum_layer(x)
            x = resonance_attn(x)
            x = self.layer_norm(x)
        
        logits = self.output_projection(x)
        return logits

# === УЛУЧШЕННЫЙ ТОКЕНИЗАТОР ===
class RussianTokenizer:
    def __init__(self):
        self.vocab = defaultdict(lambda: len(self.vocab))
        self.reverse_vocab = {}
        
        self.PAD = self.vocab['<PAD>']
        self.UNK = self.vocab['<UNK>']
        self.SOS = self.vocab['<SOS>']
        self.EOS = self.vocab['<EOS>']
        
        self._init_base_vocab()
    
    def _init_base_vocab(self):
        base_chars = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя "
        base_chars += base_chars.upper()
        base_chars += "0123456789.,!?;:-()\"'"
        
        for char in base_chars:
            self.vocab[char]
        
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
    
    def encode(self, text, max_length=120):
        tokens = [self.SOS]
        for char in text[:max_length-2]:
            tokens.append(self.vocab.get(char, self.UNK))
        tokens.append(self.EOS)
        
        while len(tokens) < max_length:
            tokens.append(self.PAD)
            
        return tokens[:max_length]
    
    def decode(self, tokens):
        text = []
        for t in tokens:
            if t == self.EOS:
                break
            if t not in [self.PAD, self.SOS]:
                text.append(self.reverse_vocab.get(t, '?'))
        return ''.join(text)

# === УЛУЧШЕННАЯ СИСТЕМА ГЕНЕРАЦИИ ===
class WaveChatSystem:
    def __init__(self):
        self.tokenizer = RussianTokenizer()
        self.vocab_size = len(self.tokenizer.vocab)
        
        self.model = WaveTransformer(
            vocab_size=self.vocab_size,
            hidden_dim=256,
            num_layers=3
        )
        
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.PAD)
        
        self.index = None
        self.corpus = None
        self.embedding_model = None
        self.knowledge_df = None
        
        print(f"✅ Система инициализирована. Размер словаря: {self.vocab_size}")

    def load_knowledge_base(self):
        print("📖 Загрузка базы знаний...")
        df = pd.read_excel(EXCEL_FILE)
        df = df.dropna(subset=['Вопрос', 'Ответ'])
        df['Вопрос'] = df['Вопрос'].astype(str).str.strip()
        df['Ответ'] = df['Ответ'].astype(str).str.strip()
        df = df[df['Ответ'].str.len() > 10]
        
        self.knowledge_df = df
        self.corpus = df['Вопрос'].tolist()
        print(f"✅ Загружено {len(df)} пар вопрос-ответ")

    def create_search_index(self):
        if self.index is not None:
            return
            
        self.load_knowledge_base()
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        
        print("📊 Создание поискового индекса...")
        embeddings = self.embedding_model.encode(self.corpus, show_progress_bar=True, convert_to_numpy=True)
        
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        
        print("✅ Поисковый индекс создан")

    def search_knowledge(self, query, top_k=3):
        if self.index is None:
            self.create_search_index()
            
        query_emb = self.embedding_model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_emb)
        
        k = min(top_k, len(self.corpus))
        distances, indices = self.index.search(query_emb, k)
        
        results = []
        for i, distance in zip(indices[0], distances[0]):
            if distance > 0.2:  # Понижен порог для большего охвата
                question = self.corpus[i]
                match = self.knowledge_df[self.knowledge_df['Вопрос'] == question]
                if not match.empty:
                    results.append(match.iloc[0]['Ответ'])
        
        return results

    def train_on_qa_pairs(self):
        if self.knowledge_df is None:
            self.load_knowledge_base()
            
        self.model.train()
        total_loss = 0
        
        # Берем больше примеров с полными ответами
        sample_df = self.knowledge_df[self.knowledge_df['Ответ'].str.len() > 50].sample(
            min(20, len(self.knowledge_df))
        )
        
        for _, row in sample_df.iterrows():
            question = row['Вопрос'][:60]
            answer = row['Ответ'][:100]  # Более длинные ответы для обучения
            
            input_tokens = self.tokenizer.encode("Вопрос: " + question)
            target_tokens = self.tokenizer.encode("Ответ: " + answer)
            
            input_tensor = torch.tensor([input_tokens], dtype=torch.long)
            target_tensor = torch.tensor([target_tokens], dtype=torch.long)
            
            self.optimizer.zero_grad()
            output = self.model(input_tensor)
            
            loss = self.criterion(output.view(-1, self.vocab_size), target_tensor.view(-1))
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(sample_df)

    def generate_complete_answer(self, question, context_answers, max_length=120, temperature=0.7):
        """Улучшенная генерация с контролем длины и качества"""
        self.model.eval()
        
        if not context_answers:
            return "Информация по вашему вопросу не найдена в базе знаний. Пожалуйста, уточните ваш вопрос."
        
        # Объединяем несколько контекстов для более полного ответа
        combined_context = " ".join(context_answers[:2])[:300]
        
        # Улучшенный промпт с явным указанием на полноту ответа
        prompt = f"Контекст: {combined_context} | Вопрос: {question} | Полный ответ:"
        
        input_tokens = self.tokenizer.encode(prompt)
        generated_tokens = []
        
        with torch.no_grad():
            for step in range(max_length):
                input_tensor = torch.tensor([input_tokens], dtype=torch.long)
                output = self.model(input_tensor)
                
                next_token_logits = output[0, -1] / temperature
                
                # Penalize repeating tokens
                for token in set(generated_tokens[-10:]):  # Смотрим последние 10 токенов
                    next_token_logits[token] -= 0.5
                
                next_token_probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(next_token_probs, 1).item()
                
                # Критерии остановки
                if next_token == self.tokenizer.EOS:
                    if len(generated_tokens) > 20:  # Минимальная длина ответа
                        break
                    else:
                        continue  # Продолжаем если ответ слишком короткий
                
                generated_tokens.append(next_token)
                input_tokens.append(next_token)
                
                # Поддерживаем длину контекста
                if len(input_tokens) > 100:
                    input_tokens = input_tokens[-100:]
                
                # Останавливаемся при завершенных предложениях
                if len(generated_tokens) > 40:
                    last_chars = self.tokenizer.decode(generated_tokens[-5:])
                    if any(mark in last_chars for mark in ['.', '!', '?', ';']):
                        # С вероятностью 30% завершаем после пунктуации
                        if torch.rand(1).item() < 0.3:
                            break
        
        response = self.tokenizer.decode(generated_tokens)
        
        # Пост-обработка ответа
        if not response.strip():
            return context_answers[0][:200]  # Fallback на оригинальный ответ
        
        # Убедимся, что ответ не обрывается на полуслове
        if len(response) > 10 and not response.endswith(('.', '!', '?', ';')):
            response += '.'
        
        return response

    def chat(self, message):
        """Улучшенная функция чата"""
        # Поиск релевантной информации
        context_answers = self.search_knowledge(message)
        
        # Определяем тип вопроса для адаптации ответа
        question_type = self.classify_question(message)
        
        if context_answers:
            response = self.generate_complete_answer(message, context_answers)
            
            # Дополнительная проверка релевантности
            if not self.is_response_relevant(response, message):
                response = context_answers[0][:250] + "..."
                
        else:
            response = "К сожалению, в моей базе знаний нет подробной информации по вашему вопросу. Рекомендую обратиться в налоговую службу или консультационный центр Приморского края для получения актуальной информации."
        
        return response

    def classify_question(self, question):
        """Классификация типа вопроса для адаптации ответа"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['что такое', 'определение', 'означает']):
            return 'definition'
        elif any(word in question_lower for word in ['как', 'инструкция', 'шаги']):
            return 'howto'
        elif any(word in question_lower for word in ['какой', 'какие', 'перечисли']):
            return 'list'
        else:
            return 'general'

    def is_response_relevant(self, response, question):
        """Проверка релевантности ответа вопросу"""
        question_words = set(re.findall(r'\b\w{3,}\b', question.lower()))
        response_words = set(re.findall(r'\b\w{3,}\b', response.lower()))
        
        common_words = question_words.intersection(response_words)
        return len(common_words) >= 2

# === ИНИЦИАЛИЗАЦИЯ ===
print("🚀 Инициализация улучшенной системы...")
chat_system = WaveChatSystem()
chat_system.create_search_index()

# Обучение на полных ответах
print("🎯 Обучение на полных ответах...")
try:
    for epoch in range(2):
        loss = chat_system.train_on_qa_pairs()
        print(f"Эпоха {epoch + 1}, Потери: {loss:.4f}")
except Exception as e:
    print(f"⚠️ Обучение пропущено: {e}")

print("✅ Система готова к работе!")

# === ИНТЕРФЕЙС ===
def save_dialog(user_msg, bot_msg):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "user": user_msg,
        "bot": bot_msg
    }
    
    try:
        with open(HISTORY_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass

def chat_interface(message, history):
    if not message.strip():
        return "", history
    
    try:
        response = chat_system.chat(message)
        save_dialog(message, response)
        
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})
        
        return "", history
        
    except Exception as e:
        error_msg = "Извините, произошла ошибка при обработке запроса. Пожалуйста, попробуйте еще раз."
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": error_msg})
        return "", history

with gr.Blocks(title="🦊 Фокстрот", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("""
    # 🦊 Фокстрот - Бизнес-советник Приморского края
    *Полные и развернутые ответы на ваши вопросы*
    """)
    
    chatbot = gr.Chatbot(
        label="Чат с Фокстротом",
        avatar_images=(None, LOGO_PATH),
        height=500,
        type="messages"
    )
    
    msg = gr.Textbox(
        placeholder="Задайте вопрос о бизнесе, налогах, ИП или ООО...",
        label="Ваш вопрос",
        max_lines=2
    )
    
    with gr.Row():
        submit_btn = gr.Button("📨 Отправить", variant="primary")
        clear_btn = gr.Button("🗑️ Очистить историю")
    
    msg.submit(chat_interface, [msg, chatbot], [msg, chatbot])
    submit_btn.click(chat_interface, [msg, chatbot], [msg, chatbot])
    clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
