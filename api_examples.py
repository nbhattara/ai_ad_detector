"""
api_examples.py
Python examples for using the API
"""
import requests
import json

# Base URL
BASE_URL = "http://localhost:8000"

# ==================== AUTHENTICATION ====================

def register_user():
    """Register a new user"""
    url = f"{BASE_URL}/auth/register"
    data = {
        "username": "testuser",
        "email": "test@example.com",
        "password": "testpass123"
    }
    
    response = requests.post(url, json=data)
    print(f"Register User: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    return response.json()


def login_user():
    """Login as user"""
    url = f"{BASE_URL}/auth/login"
    data = {
        "username": "testuser",
        "password": "testpass123"
    }
    
    response = requests.post(url, json=data)
    print(f"\nLogin User: {response.status_code}")
    result = response.json()
    print(f"Token: {result.get('access_token', 'N/A')[:50]}...")
    return result.get('access_token')


def login_admin():
    """Login as admin"""
    url = f"{BASE_URL}/auth/admin/login"
    data = {
        "username": "admin",
        "password": "admin123"
    }
    
    response = requests.post(url, json=data)
    print(f"\nAdmin Login: {response.status_code}")
    result = response.json()
    print(f"Admin Token: {result.get('access_token', 'N/A')[:50]}...")
    return result.get('access_token')


# ==================== DETECTION ====================

def detect_with_ml(token=None):
    """ML-based detection"""
    url = f"{BASE_URL}/detect/ml"
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    
    # Ad-like features
    data = {
        "click_through_rate": 6.5,
        "engagement_rate": 1.8,
        "time_on_page": 12.3,
        "image_count": 6,
        "link_count": 12,
        "caps_ratio": 0.32,
        "exclamation_count": 5,
        "special_offer": 1,
        "urgency_words": 1,
        "price_mentioned": 1
    }
    
    response = requests.post(url, json=data, headers=headers)
    print(f"\nML Detection: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    return response.json()


def detect_with_nlp(token=None):
    """NLP-based detection"""
    url = f"{BASE_URL}/detect/text"
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    
    # Ad text
    data = {
        "text": "LIMITED TIME OFFER! Get 70% OFF on all products! Shop now and save BIG! FREE shipping! Don't miss out!"
    }
    
    response = requests.post(url, json=data, headers=headers)
    print(f"\nNLP Detection (Ad Text): {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    
    # Non-ad text
    data = {
        "text": "The research paper discusses the methodology and findings of the comprehensive study conducted over a period of three years."
    }
    
    response = requests.post(url, json=data, headers=headers)
    print(f"\nNLP Detection (Non-Ad Text): {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    
    return response.json()


def detect_with_lstm(token=None):
    """LSTM-based detection"""
    url = f"{BASE_URL}/detect/lstm"
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    
    data = {
        "text": "EXCLUSIVE DEAL! Save $100 today! Order now and get FREE bonus gifts! Limited stock!"
    }
    
    response = requests.post(url, json=data, headers=headers)
    print(f"\nLSTM Detection: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    return response.json()


# ==================== ADMIN ====================

def get_dashboard_stats(admin_token):
    """Get dashboard statistics"""
    url = f"{BASE_URL}/admin/stats"
    headers = {"Authorization": f"Bearer {admin_token}"}
    
    response = requests.get(url, headers=headers)
    print(f"\nDashboard Stats: {response.status_code}")
    stats = response.json()
    print(f"Total Users: {stats.get('total_users')}")
    print(f"Total Detections: {stats.get('total_detections')}")
    print(f"Ads Detected: {stats.get('total_ads_detected')}")
    print(f"Recent Logs: {len(stats.get('recent_logs', []))}")
    return stats


# ==================== MAIN DEMO ====================

def main():
    """Run complete API demo"""
    print("="*60)
    print("  AI ADVERTISEMENT DETECTION PLATFORM - API DEMO")
    print("="*60)
    
    try:
        # 1. Register user (might fail if already exists)
        print("\n[1] REGISTERING USER...")
        try:
            register_user()
        except Exception as e:
            print(f"Registration skipped (user may already exist): {e}")
        
        # 2. Login as user
        print("\n[2] LOGGING IN AS USER...")
        user_token = login_user()
        
        # 3. Test ML detection
        print("\n[3] TESTING ML DETECTION...")
        detect_with_ml(user_token)
        
        # 4. Test NLP detection
        print("\n[4] TESTING NLP DETECTION...")
        detect_with_nlp(user_token)
        
        # 5. Test LSTM detection
        print("\n[5] TESTING LSTM DETECTION...")
        detect_with_lstm(user_token)
        
        # 6. Login as admin
        print("\n[6] LOGGING IN AS ADMIN...")
        admin_token = login_admin()
        
        # 7. Get dashboard stats
        print("\n[7] FETCHING DASHBOARD STATS...")
        get_dashboard_stats(admin_token)
        
        print("\n" + "="*60)
        print("  ✓ ALL API TESTS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nNext steps:")
        print("  1. Visit http://localhost:8000/docs for interactive API docs")
        print("  2. Access admin dashboard with admin token")
        print("  3. Explore the codebase and customize!")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Cannot connect to API server")
        print("Please ensure the server is running:")
        print("  python app/main.py")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()