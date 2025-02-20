

def check_and_install_packages(package_list):

# Function to check for required packages and install them if not found
# integrates functionality to operate in Jupyter environment (including 
# Google Colab) or standard IDE (VSCode, PyCharm, etc.)
# ==================================================================== #
    for package in package_list:
        try:
            importlib.import_module(package)
        except ImportError:
            print(f"{package} not found, installing...")
            try:
                # Check if in a Jupyter (IPython) environment
                if 'get_ipython' in globals():
                    print("Using Jupyter magic command to install.")
                    get_ipython().system(f'pip install {package}')
                else:
                    # Fallback to standard IDE installation method
                    subprocess.run(
                        [sys.executable, '-m', 'pip', 'install', package], 
                        check=True, 
                        capture_output=True
                        )
            except Exception as e:
                print(f"{package} not installed: {e}")
            # Try importing the package again after installation
            importlib.import_module(package)


