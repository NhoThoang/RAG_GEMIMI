import google.generativeai as genai

# Thay bằng API key thật của bạn
API_KEY = "AIzaSyCXquGeTVFrXUxPGypwzzWS4AKbwd4RcAs"
genai.configure(api_key=API_KEY)

def test_gemini_api():
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content("Hi Gemini 2.0! Can you confirm this API key works?")
        print("✅ API hoạt động. Phản hồi từ Gemini:")
        print(response.text)
    except Exception as e:
        print("❌ API key lỗi hoặc không truy cập được Gemini:")
        print(e)

if __name__ == "__main__":
    test_gemini_api()
    print("✅ Kết quả test API key:")