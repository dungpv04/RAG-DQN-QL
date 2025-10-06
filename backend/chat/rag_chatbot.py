import re
import json
import numpy as np
from algorithm.q_learning import QLearningAgent

class RAGChatbot:
    def __init__(self, agent, env, preprocessed_data=None):
        self.agent = agent
        self.env = env
        self.documents = self._load_preprocessed_data(preprocessed_data)

    def _load_preprocessed_data(self, data):
        """Load dữ liệu đã được chunk sẵn"""
        if data is not None:
            print(f"Loaded {len(data)} preprocessed chunks")
            return data
        
        # Fallback: thử đọc từ file JSON nếu có
        try:
            with open('data/processed_chunks.json', 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            print(f"Loaded {len(chunks)} chunks from JSON file")
            return chunks
        except:
            print("Using fallback content")
            return self._get_fallback_chunks()
    
    def _get_fallback_chunks(self):
        """Dữ liệu fallback dạng chunks"""
        return [
            {
                "title": "QUY CHẾ ĐÀO TẠO TRÌNH ĐỘ ĐẠI HỌC", 
                "content": "Quy chế này quy định về tổ chức và quản lý đào tạo trình độ đại học",
                "source": "fallback"
            },
            {
                "title": "Thời gian đào tạo", 
                "content": "Thời gian chuẩn: Chính quy 4,0-4,5 năm, Vừa làm vừa học 4,5-5,0 năm. Thời gian tối đa: Chính quy 8,0-9,0 năm",
                "source": "fallback"
            }
        ]
        
    def respond(self, query):
        """Trả lời câu hỏi của sinh viên"""
        # Tìm kiếm chunks liên quan
        relevant_chunks = self._search_relevant_chunks(query)
        
        # Trích xuất đặc trưng từ chunks
        chunk_contents = [chunk['content'] for chunk in relevant_chunks]
        features = self.env.extract_features(query, chunk_contents)
        state = self.env.encode_state(features)
        
        print(f"Debug - Query: {query}")
        print(f"Debug - Features: {features}")
        print(f"Debug - Found {len(relevant_chunks)} relevant chunks")
        
        # Chọn hành động tốt nhất
        if isinstance(self.agent, QLearningAgent):
            action = np.argmax(self.agent.q_table[state])
        else:  # DQN
            action = self.agent.act(state)
        
        action_name = self.env.actions[action]
        
        # Tạo response từ chunks
        response = self._generate_response(action, query, relevant_chunks)
        
        return response, action_name
    
    def _search_relevant_chunks(self, query, top_k=3):
        """Tìm kiếm chunks liên quan nhất với query"""
        query_words = set(self._normalize_text(query).split())
        scored_chunks = []
        
        for chunk in self.documents:
            # Tính điểm cho title và content
            title_words = set(self._normalize_text(chunk['title']).split())
            content_words = set(self._normalize_text(chunk['content']).split())
            
            # Điểm cho title (trọng số cao hơn)
            title_score = len(query_words & title_words) * 3
            
            # Điểm cho content
            content_common = len(query_words & content_words)
            content_ratio = content_common / max(len(query_words), 1)
            content_score = content_common * 2 + content_ratio * 5
            
            # Điểm cho độ dài content (ưu tiên content có thông tin)
            length_score = min(len(chunk['content'].split()) / 30, 2.0)
            
            # Tổng điểm
            total_score = title_score + content_score + length_score
            
            if total_score > 0:
                scored_chunks.append((chunk, total_score))
        
        # Sắp xếp theo điểm và lấy top_k
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return [chunk[0] for chunk in scored_chunks[:top_k]]
    
    def _generate_response(self, action, query, chunks):
        """Tạo response dựa trên action và chunks"""
        if not chunks:
            return "Xin lỗi, tôi không tìm thấy thông tin liên quan trong quy chế."
        
        if action == 0:  # Trích dẫn nguyên văn
            quote = self._extract_best_quote(query, chunks)
            return f"Theo quy chế: \"{quote}\""
            
        elif action == 1:  # Tóm tắt nội dung
            summary = self._create_summary(query, chunks)
            return f"Tóm tắt: {summary}"
            
        elif action == 2:  # Diễn giải lại
            explanation = self._create_explanation(query, chunks)
            return f"Có thể hiểu rằng: {explanation}"
            
        elif action == 3:  # Hỏi lại để làm rõ
            return "Bạn có thể làm rõ thêm câu hỏi được không? Tôi cần thông tin cụ thể hơn."
            
        else:  # Thoái lui an toàn
            return "Xin lỗi, câu hỏi này nằm ngoài phạm vi quy chế đào tạo."
    
    def _extract_best_quote(self, query, chunks):
        """Trích xuất câu trích dẫn tốt nhất từ chunks"""
        query_words = set(self._normalize_text(query).split())
        best_quote = ""
        max_relevance = 0
        
        for chunk in chunks:
            # Kiểm tra title trước
            title_words = set(self._normalize_text(chunk['title']).split())
            title_relevance = len(query_words & title_words)
            
            if title_relevance > 0:
                if title_relevance > max_relevance:
                    max_relevance = title_relevance
                    best_quote = chunk['title']
            
            # Sau đó kiểm tra content
            sentences = self._split_sentences(chunk['content'])
            for sentence in sentences:
                if len(sentence.split()) < 5:
                    continue
                
                sentence_words = set(self._normalize_text(sentence).split())
                relevance = len(query_words & sentence_words)
                
                if relevance > max_relevance:
                    max_relevance = relevance
                    best_quote = sentence.strip()
        
        # Rút gọn nếu quá dài
        if len(best_quote) > 200:
            best_quote = best_quote[:200] + "..."
            
        return best_quote if best_quote else "Thông tin được quy định trong quy chế đào tạo."
    
    def _create_summary(self, query, chunks):
        """Tạo tóm tắt từ các chunks liên quan"""
        query_words = set(self._normalize_text(query).split())
        summary_parts = []
        
        for chunk in chunks:
            # Thêm title nếu liên quan
            title_words = set(self._normalize_text(chunk['title']).split())
            if len(query_words & title_words) > 0:
                summary_parts.append(f"{chunk['title']}: ")
            
            # Tìm câu liên quan trong content
            sentences = self._split_sentences(chunk['content'])
            for sentence in sentences:
                if len(sentence.split()) > 5:
                    sentence_words = set(self._normalize_text(sentence).split())
                    if len(query_words & sentence_words) > 0:
                        summary_parts.append(sentence.strip())
                        break  # Chỉ lấy 1 câu quan trọng nhất mỗi chunk
        
        summary_text = ". ".join(summary_parts[:3])  # Tối đa 3 phần
        
        if len(summary_text) > 300:
            summary_text = summary_text[:300] + "..."
            
        return summary_text if summary_text else "Các quy định liên quan được nêu trong quy chế đào tạo."
    
    def _create_explanation(self, query, chunks):
        """Tạo diễn giải từ chunks"""
        summary = self._create_summary(query, chunks)
        
        # Thêm ngữ cảnh giải thích
        if "điều kiện" in query.lower() or "yêu cầu" in query.lower():
            return f"theo quy định, các yêu cầu như sau: {summary.lower()}"
        elif "thời gian" in query.lower():
            return f"về thời gian, quy chế quy định: {summary.lower()}"
        elif "học phí" in query.lower() or "tín chỉ" in query.lower():
            return f"liên quan đến vấn đề này, {summary.lower()}"
        else:
            return f"theo quy chế, {summary.lower()}"
    
    def _split_sentences(self, text):
        """Chia văn bản thành các câu"""
        # Pattern đơn giản cho tiếng Việt
        sentences = re.split(r'[.!?;]\s*', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _normalize_text(self, text):
        """Chuẩn hóa text để so sánh"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

# Utility function để load data
def load_chunked_data(json_file_path=None, data_list=None):
    """Helper function để load dữ liệu chunks"""
    if data_list is not None:
        return data_list
    
    if json_file_path:
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading JSON: {e}")
    
    return None

# Example usage:
"""
# Cách sử dụng với dữ liệu chunks sẵn có:
chunked_data = [
    {
        "title": "Điều 1. Phạm vi điều chỉnh", 
        "content": "Quy chế này quy định về tổ chức và quản lý đào tạo...",
        "source": "321"
    },
    # ... more chunks
]

# Khởi tạo chatbot
chatbot = RAGChatbot(trained_agent, env, chunked_data)

# Hoặc từ file JSON:
chatbot = RAGChatbot(trained_agent, env, load_chunked_data('data/chunks.json'))

# Sử dụng
response, action = chatbot.respond("Học phí được tính như thế nào?")
"""