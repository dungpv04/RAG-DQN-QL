from algorithm.q_learning import QLearningAgent
import numpy as np
from helper.helper_functions import get_retrieved_text

class RAGChatbot:
    def __init__(self, agent, env):
        self.agent = agent
        self.env = env
        self.documents = self._prepare_documents()

    def load_documents(self):
        """Load documents from the notebook content"""
        with open('data/notebook_content.txt', 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"Loaded content: f{content}")
        # Split content into meaningful chunks
        return content
        
    def respond(self, query):
        """Trả lời câu hỏi của sinh viên"""
        # Giả định truy xuất tài liệu
        retrieved_docs = self.documents
        
        # Trích xuất đặc trưng
        features = self.env.extract_features(query, retrieved_docs)
        state = self.env.encode_state(features)
        
        # Chọn hành động tốt nhất
        if isinstance(self.agent, QLearningAgent):
            action = np.argmax(self.agent.q_table[state])
        else:  # DQN
            action = self.agent.act(state)
        
        action_name = self.env.actions[action]
        
        # Thực hiện hành động tương ứng
        if action == 0:  # Trích dẫn nguyên văn
            response = f"Theo tài liệu: {get_retrieved_text(retrieved_docs[:1])}"
        elif action == 1:  # Tóm tắt
            response = f"Tóm tắt: {get_retrieved_text(retrieved_docs)[:100]}..."
        elif action == 2:  # Diễn giải
            response = f"Có thể hiểu rằng: {get_retrieved_text(retrieved_docs)[:80]}..."
        elif action == 3:  # Hỏi lại
            response = "Bạn có thể làm rõ thêm câu hỏi được không?"
        else:  # Thoái lui
            response = "Xin lỗi, tôi chưa có đủ thông tin để trả lời câu hỏi này."
        
        return response, action_name
    
    def _prepare_documents(self):
        documents = self.load_documents()
        cleaned_chunks = []
        for doc in documents:
            cleaned_doc = self._clean_text(doc)
            chunks = self._chunk_text(cleaned_doc)
            cleaned_chunks.extend(chunks)
        return cleaned_chunks
        
    def _clean_text(self, text):
        import re
        text = re.sub(r'[\n\r]+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _chunk_text(self, text, chunk_size=500, overlap=50):
        chunks = []
        words = text.split()
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks