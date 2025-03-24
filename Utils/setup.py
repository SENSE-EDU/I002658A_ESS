import subprocess
import importlib
import sys

import importlib
import subprocess
import sys

def check_and_install_packages(package_list):
    """
    Checks for required packages and installs them if not found.
    Works in both Jupyter environments (e.g. Google Colab) and standard IDEs (e.g. VSCode, PyCharm).

    Parameters:
    -----------
    package_list : list of str
        List of package/module names to check and install if necessary.

    Returns:
    --------
    None
    """

    # Mapping from import name to actual pip install name (only needed for edge cases)
    pip_name_map = {
        'sklearn': 'scikit-learn'
        # Add more mappings here if needed
    }

    for package in package_list:
        pip_name = pip_name_map.get(package, package)

        try:
            importlib.import_module(package)
            print(f"✅ {package} is already installed.")
        except ImportError:
            print(f"📦 {package} not found. Installing '{pip_name}'...")

            try:
                if 'get_ipython' in globals():
                    print("🔧 Detected Jupyter environment. Using magic command...")
                    get_ipython().system(f'pip install {pip_name}')
                else:
                    print("🔧 Using subprocess pip install...")
                    subprocess.run(
                        [sys.executable, '-m', 'pip', 'install', pip_name],
                        check=True
                    )
            except Exception as install_error:
                print(f"❌ Failed to install {pip_name}: {install_error}")
                continue

            # Try importing again
            try:
                importlib.import_module(package)
                print(f"✅ {package} installed and imported successfully.")
            except ImportError as e:
                print(f"❌ Installation succeeded but failed to import {package}: {e}")


# def check_and_install_packages(package_list):
#     """
#     Checks for required packages and installs them if not found.
#     This function integrates functionality to operate in Jupyter environments (including 
#     Google Colab) or standard IDEs (VSCode, PyCharm, etc.).
#     Parameters:
#     -----------
#     package_list : list
#         A list of package names (as strings) to check and install if necessary.
#     Returns:
#     --------
#     None
#     """
#     for package in package_list:
#         try:
#             importlib.import_module(package)
#         except ImportError:
#             print(f"{package} not found, installing...")
#             try:
#                 # Check if in a Jupyter (IPython) environment
#                 if 'get_ipython' in globals():
#                     print("Using Jupyter magic command to install.")
#                     get_ipython().system(f'pip install {package}')
#                 else:
#                     # Fallback to standard IDE installation method
#                     subprocess.run(
#                         [sys.executable, '-m', 'pip', 'install', package], 
#                         check=True, 
#                         capture_output=True
#                     )
#             except Exception as e:
#                 print(f"{package} not installed: {e}")
#             # Try importing the package again after installation
#             try:
#                 importlib.import_module(package)
#             except ImportError as e:
#                print(f"Failed to import {package} after installation: {e}")