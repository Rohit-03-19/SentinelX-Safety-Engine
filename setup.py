from setuptools import setup, find_packages

setup(
    name="SentinelX_Safety_Engine",
    version="0.0.1",
    author="Rohit",
    # find_packages() ki jagah list de do agar wo kaam nahi kar raha
    packages=find_packages(), 
    install_requires=[], # Humne requirements.txt se install kar liya hai
)