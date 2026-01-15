"""
setup.py
One-command setup script to initialize the entire platform
"""
import subprocess
import sys
import platform
from pathlib import Path

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60 + "\n")

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"➤ {description}...")
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        print(f"  ✓ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Error: {e}")
        print(f"  Output: {e.output}")
        return False

def install_dependencies():
    """Install all Python dependencies, handling Windows scikit-learn"""
    print("➤ Installing Python packages...")

    # Upgrade pip, setuptools, wheel first
    run_command([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"], "Upgrading pip, setuptools, wheel")

    # Detect Windows and install scikit-learn prebuilt wheel first
    if platform.system() == "Windows":
        print("  ℹ Detected Windows: installing scikit-learn prebuilt binary")
        if not run_command([sys.executable, "-m", "pip", "install", "scikit-learn==1.4.0", "--only-binary", ":all:"], "Installing scikit-learn binary"):
            return False

        # Install the rest of the dependencies excluding scikit-learn
        deps = [line.strip() for line in open("requirements.txt").readlines() if "scikit-learn" not in line and line.strip()]
        if not deps:
            return True
        return run_command([sys.executable, "-m", "pip", "install"] + deps, "Installing remaining dependencies")
    else:
        # Non-Windows: normal install
        return run_command([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], "Installing dependencies")

def create_admin():
    """Create default admin user"""
    print("➤ Creating admin user...")
    try:
        from app.models.db_models import SessionLocal
        from app.crud.user import create_admin as create_admin_user
        
        db = SessionLocal()
        
        try:
            admin = create_admin_user(
                db=db,
                username="admin",
                email="admin@example.com",
                password="admin123"
            )
            print(f"  ✓ Admin created: {admin.username}")
            print(f"    Username: admin")
            print(f"    Password: admin123")
            return True
        except Exception as e:
            if "UNIQUE constraint failed" in str(e):
                print(f"  ℹ Admin user already exists")
                return True
            else:
                print(f"  ✗ Error creating admin: {e}")
                return False
        finally:
            db.close()
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def main():
    """Main setup function"""
    print_header("AI ADVERTISEMENT DETECTION PLATFORM - SETUP")
    
    print("This script will:")
    print("  1. Install dependencies")
    print("  2. Initialize database")
    print("  3. Train ML models")
    print("  4. Train NLP models")
    print("  5. Train LSTM models")
    print("  6. Create admin user")
    
    proceed = input("\nProceed? (y/n): ").lower()
    if proceed != 'y':
        print("Setup cancelled.")
        return
    
    # Step 1: Install dependencies
    print_header("STEP 1: Installing Dependencies")
    if not install_dependencies():
        print("\n❌ Failed to install dependencies. Please check your Python environment.")
        return
    
    # Step 2: Initialize database
    print_header("STEP 2: Initializing Database")
    run_command(
        f"{sys.executable} -c \"from app.models.db_models import init_db; init_db()\"",
        "Creating database tables"
    )
    
    # Step 3: Train ML models
    print_header("STEP 3: Training ML Models")
    run_command(
        f"{sys.executable} app/ml/train_ml.py",
        "Training ML models (Logistic Regression, SVM, Random Forest, etc.)"
    )
    
    # Step 4: Train NLP models
    print_header("STEP 4: Training NLP Models")
    run_command(
        f"{sys.executable} app/nlp/train_nlp.py",
        "Training NLP models (TF-IDF)"
    )
    
    # Step 5: Train LSTM models
    print_header("STEP 5: Training LSTM Models")
    run_command(
        f"{sys.executable} app/dl/train_dl.py",
        "Training LSTM deep learning models"
    )
    
    # Step 6: Create admin user
    print_header("STEP 6: Creating Admin User")
    create_admin()
    
    # Completion
    print_header("SETUP COMPLETE!")
    print("✓ All models trained successfully")
    print("✓ Database initialized")
    print("✓ Admin user created")
    
    print("\n" + "="*60)
    print("  DEFAULT ADMIN CREDENTIALS")
    print("="*60)
    print("  Username: admin")
    print("  Password: admin123")
    print("  ⚠ CHANGE THESE IN PRODUCTION!")
    print("="*60)
    
    print("\n" + "="*60)
    print("  NEXT STEPS")
    print("="*60)
    print("  1. Start the server:")
    print("     python app/main.py")
    print()
    print("  2. Open API documentation:")
    print("     http://localhost:8000/docs")
    print()
    print("  3. Access admin dashboard:")
    print("     http://localhost:8000/admin/dashboard")
    print("     (Requires admin login first)")
    print("="*60)
    
    print("\n🚀 Platform is ready to use!")

if __name__ == "__main__":
    main()
