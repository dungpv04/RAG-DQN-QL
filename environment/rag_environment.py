import numpy as np
import random
from helper.helper_functions import embedding_function, rerank_function

class RAGEnvironment:
    def __init__(self):
        # Định nghĩa MDP
        self.n_states = 3**5  # 243 trạng thái (5 đặc trưng, mỗi đặc trưng 3 mức)
        self.n_actions = 5    # 5 hành động
        self.gamma = 0.95     # Hệ số chiết khấu
        
        # Hành động và phần thưởng
        self.actions = {
            0: "trích dẫn nguyên văn",  
            1: "tóm tắt nội dung", 
            2: "diễn giải lại",
            3: "hỏi lại để làm rõ",
            4: "thoái lui an toàn"
        }
        
        self.current_state = 0
        
    def encode_state(self, features):
        """Mã hóa 5 đặc trưng thành trạng thái số"""
        state = 0
        for i, feature in enumerate(features):
            state += feature * (3 ** i)
        return state
    
    def decode_state(self, state):
        """Giải mã trạng thái số thành 5 đặc trưng"""
        features = []
        for i in range(5):
            features.append(state % 3)
            state //= 3
        return features
    
    def extract_features(self, query, retrieved_docs):
        """Trích xuất 5 đặc trưng từ câu hỏi và tài liệu truy xuất"""
        # Giả định các hàm phụ trợ đã tồn tại
        similarity = self._get_similarity_level(query, retrieved_docs)
        confidence = self._get_retrieval_confidence(retrieved_docs)
        context = self._get_context_match(query, retrieved_docs)
        ambiguity = self._get_ambiguity_level(query)
        scope = self._get_scope_match(query, retrieved_docs)
        
        return [similarity, confidence, context, ambiguity, scope]
    
    def _get_similarity_level(self, query, docs):
        """Mức độ tương đồng ngữ nghĩa (0: thấp, 1: trung bình, 2: cao)"""
        # Giả định sử dụng cosine similarity
        scores = [embedding_function(query, doc) for doc in docs]
        avg_score = np.mean(scores) if scores else 0
        if avg_score > 0.8: return 2
        elif avg_score > 0.5: return 1
        else: return 0
    
    def _get_retrieval_confidence(self, docs):
        """Độ tin cậy truy hồi dựa trên rerank score"""
        if not docs: return 0
        scores = [rerank_function(doc) for doc in docs]
        max_score = max(scores)
        if max_score > 0.9: return 2
        elif max_score > 0.6: return 1
        else: return 0
    
    def _get_context_match(self, query, docs):
        """Đánh giá mức độ phù hợp ngữ cảnh giữa query và các tài liệu"""
        if not docs:
            return 0  # không có tài liệu thì coi như không khớp

        # Tính độ tương đồng trung bình giữa query và tất cả tài liệu
        sims = [embedding_function(query, doc) for doc in docs]
        avg_sim = np.mean(sims)

        # Quy tắc ánh xạ về 3 mức
        if avg_sim > 0.75:
            return 2  # Ngữ cảnh khớp tốt
        elif avg_sim > 0.4:
            return 1  # Ngữ cảnh khớp trung bình
        else:
            return 0  # Ngữ cảnh kém
    
    def _get_ambiguity_level(self, query):
        """Độ mơ hồ trong câu hỏi"""
        ambiguous_words = ["có thể", "như thế nào", "gì", "tại sao"]
        count = sum(1 for word in ambiguous_words if word in query.lower())
        if count >= 2: return 2
        elif count == 1: return 1
        else: return 0
    
    def _get_scope_match(self, query, docs):
        """Đánh giá mức độ phù hợp phạm vi giữa query và docs."""
        if not docs:
            return 0  # không có tài liệu thì coi như không khớp

        sims = [embedding_function(query, doc) for doc in docs]
        avg_sim = np.mean(sims)

        # Quy ra mức 0-1-2
        if avg_sim > 0.8:
            return 2  # khớp phạm vi tốt
        elif avg_sim > 0.5:
            return 1  # khớp trung bình
        else:
            return 0  # phạm vi kém khớp
    
    def get_reward(self, state, action):
        """Hàm phần thưởng dựa trên trạng thái và hành động"""
        features = self.decode_state(state)
        similarity, confidence, context, ambiguity, scope = features
        
        # Logic phần thưởng phức tạp
        if action == 0:  # Trích dẫn nguyên văn
            if similarity == 2 and confidence == 2: return 3
            elif similarity >= 1 and confidence >= 1: return 2
            else: return -1
            
        elif action == 1:  # Tóm tắt nội dung
            if confidence >= 1 and context >= 1: return 2
            elif confidence == 0: return -5
            else: return 1
            
        elif action == 2:  # Diễn giải lại
            if ambiguity >= 1 and scope >= 1: return 2
            else: return -1
            
        elif action == 3:  # Hỏi lại để làm rõ
            if ambiguity == 2: return 3
            elif ambiguity == 1: return 1
            else: return -8
            
        elif action == 4:  # Thoái lui an toàn
            if confidence == 0 and similarity == 0: return 1
            else: return -12
            
        return 0
    
    def step(self, action):
        """Thực hiện hành động và trả về trạng thái mới, phần thưởng"""
        # Tính phần thưởng tại state hiện tại
        reward = self.get_reward(self.current_state, action)

        # Giải mã state hiện tại ra 5 đặc trưng
        features = self.decode_state(self.current_state)
        similarity, confidence, context, ambiguity, scope = features

        # Quy tắc cập nhật đặc trưng theo action
        if action == 0:  # Trích dẫn nguyên văn
            similarity = min(similarity + 1, 2)  
        elif action == 1:  # Tóm tắt nội dung
            context = min(context + 1, 2)
        elif action == 2:  # Diễn giải lại
            ambiguity = max(ambiguity - 1, 0)
        elif action == 3:  # Hỏi lại
            ambiguity = min(ambiguity + 1, 2)
        elif action == 4:  # Thoái lui an toàn
            similarity, confidence, context, ambiguity, scope = 0, 0, 0, 0, 0

        # Mã hóa lại thành state mới
        new_state = self.encode_state([similarity, confidence, context, ambiguity, scope])
        self.current_state = new_state

        # Giả định chưa có điều kiện kết thúc (done)
        done = False
        return self.current_state, reward, done
    
    def reset(self):
        """Reset môi trường về trạng thái ban đầu"""
        self.current_state = random.randint(0, self.n_states - 1)
        return self.current_state