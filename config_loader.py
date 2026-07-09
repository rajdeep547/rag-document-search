# backend/config_loader.py
"""
Configuration loader with easy access to settings
"""
from backend.config import Config, ConfigSingleton

# Singleton instance
config = ConfigSingleton()

# Convenience functions
def get_config():
    """Get configuration instance"""
    return config

def get_setting(key, default=None):
    """Get a specific setting"""
    return getattr(config, key, default)

def update_setting(key, value):
    """Update a setting at runtime"""
    setattr(config, key, value)
    print(f"✅ Updated {key} = {value}")

def show_config():
    """Display current configuration"""
    print("\n" + "="*60)
    print("📊 CURRENT CONFIGURATION")
    print("="*60)
    
    summary = Config.get_summary()
    for key, value in summary.items():
        print(f"   {key}: {value}")
    
    print("="*60 + "\n")

# Auto-export on first import
if __name__ == "__main__":
    show_config()