"""
This module contains utility functions to be used in Jupyter notebooks or standard IDEs 
like VSCode, PyCharm, etc. The functions provided here help in managing package dependencies 
by checking for required packages and installing them if they are not found.
Functions:
-----------
- check_and_install_packages(package_list):
    Checks for required packages and installs them if not found. This function integrates 
    functionality to operate in Jupyter environments (including Google Colab) or standard IDEs.
"""

def check_and_install_packages(package_list):
    """
    Checks for required packages and installs them if not found.
    This function integrates functionality to operate in Jupyter environments (including 
    Google Colab) or standard IDEs (VSCode, PyCharm, etc.).
    Parameters:
    -----------
    package_list : list
        A list of package names (as strings) to check and install if necessary.
    Returns:
    --------
    None
    """
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


    def interpolate(x, y, z, cell_size, method='nearest', smooth_s=0, blank=blank):
        """
        Interpolate scatter data to regular grid through selected interpolation 
        method (with scipy.interpolate for simple interpolation).

        The output of this function is a Numpy array that holds the interpolated 
        data (accessed via `datagrid['grid']` in the cell below), alongside 
        the grid's cell size (`datagrid['cell_size']`) and the grid extent 
        (`datagrid['extent']`). The cell size and extent are needed to allow 
        exporting the interpolated data efficiently to a GeoTIF that can be 
        opened in any GIS software such as QGIS. 

        Parameters
        ----------
        x : np.array
            Cartesian GPS x-coordinates.

        y : np.array
            Cartesian GPS y-coordinates.

        z : np.array
            Data points to interpolate.

        cell_size : float
            Grid cell size (m).

        method : str, optional
            Scipy interpolation method ('nearest', 'linear', 'cubic' or 'IDW')

        smooth_s : float, optional
            Smoothing factor to apply a Gaussian filter on the interpolated grid.
            If 0, no smoothing is performed. (Applying smoothing can result 
            in a loss of detail in the interpolated grid.)

        blank : object
            A blank object to mask (clip )interpolation beyond survey bounds.

        Returns
        -------
        grid : np.array
            Array of interpolated and masked grid containing:
            - the interpolated grid (grid['grid'])
            - the grid cell size (grid['cell_size'])
            - the grid extent (grid['extent'])

        """
        x_min = x.min()
        x_max = x.max() + cell_size
        y_min = y.min()
        y_max = y.max() + cell_size
        x_vector = np.arange(x_min, x_max, cell_size)
        y_vector = np.arange(y_min, y_max, cell_size)
        extent = (x_vector[0], x_vector[-1], y_vector[0], y_vector[-1])

        xx, yy = np.meshgrid(x_vector, y_vector)
        nx, ny = xx.shape
        coords = np.concatenate((xx.ravel()[np.newaxis].T, 
                            yy.ravel()[np.newaxis].T), 
                            axis=1)

        boolean = np.zeros_like(xx)
        boundaries = np.vstack(blank.loc[0, 'geometry'].exterior.coords.xy).T
        bound = boundaries.copy()
        boolean += matplotlib.path.Path(
            bound).contains_points(coords).reshape((nx, ny))
        boolean = np.where(boolean >= 1, True, False)
        mask = np.where(boolean == False, np.nan, 1)
        binary = np.where(boolean == False, 0, 1)
        
        if method in ['nearest','cubic', 'linear']:
            data_grid = griddata(
                np.vstack((x, y)).T, z, (xx, yy), method=method
                ) * mask
        else:
            print('define interpolation method')
        
        if smooth_s > 0:
            data_grid = gaussian_filter(data_grid, sigma=smooth_s)

        dtype = [
            ('grid', data_grid.dtype, data_grid.shape),
            ('cell_size', float),
            ('extent', [
                ('x_min', float), 
                ('x_max', float), 
                ('y_min', float), 
                ('y_max', float)
                ])
        ]
        grid = np.array((data_grid, cell_size, extent), dtype=dtype)
        
        return grid


    def export_grid(grid_in, filename='georaster'): 
        """
        Interpolate scatter data to regular grid through selected interpolation 
        method (with scipy.interpolate for simple interpolation).

        Parameters
        ----------
        grid_in : np.array
            Array of interpolated and masked grid.

        filename : str, optional
            Name of the GeoTIFF (.tif) file (standard = 'gridded').

        Returns
        -------
        None
        !!! data are exported to the working directory as a GeoTIFF file.
        
        """
        cell_size = grid_in['cell_size']
        extent = grid_in['extent']
        transform = from_origin(extent['x_min'], extent['y_min'], 
                                    cell_size, -cell_size)

        grid_exp = grid_in['grid']
        grid_exp[np.isnan(grid_exp)] = -99999
        grid_exp = grid_exp.astype(rasterio.float32)
        nx, ny = grid_exp.shape
        grid_exp = np.flip(grid_exp, axis=0)

        with rasterio.open(
            filename + '.tif',
            mode='w',
            driver='GTiff',
            height=nx,
            width=ny,
            count=1,
            dtype=str(grid_exp.dtype),
            crs='EPSG:31370',
            transform=transform,
            nodata=-99999
        ) as dst:
            dst.write(grid_exp, 1)

        with rasterio.open(filename + '.tif', mode='r+') as dst:
            data = dst.read()
            dst.write(data[0, ::-1], 1)


    def lin_sens(geometry, maxdepth=3., n_int=100, sensor_height = 0):
        """
        Calculate approximative cumulative and relative sensitivities based on
        Keller & Frischknecht, 1966 and McNeill, 1980

        Parameters
        ----------
        geometry : str
            coil geometry identifier, combining orientation ('HCP' or 'PRP'),
            and Tx-Rx separation. 'HCP0.5'is thus a HCP orientation with a 0.5 m
            coil separation.

        maxdepth : float (optional, default=3.0)
            Maximum depth to evaluate the sensitivity (m).

        n_int : int (optional, default=100)
            Number of depth intervals to evaluate the sensitivity.

        sensor_height : float (optional, default=0)
            Height of the sensor above the ground (m).

        Returns
        -------
        rsens_QP : np.array
            Array of relative QP sensitivities obtained over the evaluated depths.
        
        csens_QP : np.array, optional
            Array of cumulative QP sensitivities obtained over the evaluated depths.

        rsens_IP : np.array, optional
            Array of relative IP sensitivities obtained over the evaluated depths.
        
        csens_IP : np.array, optional
            Array of cumulative IP sensitivities obtained over the evaluated depths.

        """
        depths = np.linspace(.0, maxdepth, n_int)

        if 'inph' in geometry:
            coil_spacing = float(geometry[3:6])
        else: 
            coil_spacing = float(geometry[-3:])

        csens_QP = np.empty_like(depths)
        rsens_QP = np.empty_like(depths)
        csens_IP = np.empty_like(depths)
        rsens_IP = np.empty_like(depths)
        
        depth_ratio = (depths + sensor_height) / coil_spacing
        if 'HCP' in geometry:
            csens_IP = (1 - 8 * depth_ratio ** 2) / ((4 * (depth_ratio ** 2) + 1) ** (5 / 2))
            rsens_IP = 12 * depth_ratio * (3 - 8 * depth_ratio ** 2) / (coil_spacing * ((4 * depth_ratio ** 2) + 1) ** (7 / 2))
            csens_QP = 1 / ((4 * (depth_ratio ** 2) + 1) ** 0.5)
            rsens_QP = 4 * depth_ratio / (coil_spacing * (4 * depth_ratio ** 2 + 1) ** (3 / 2))

        if 'PRP' in geometry:
            csens_IP = (6 * depth_ratio) / ((4 * (depth_ratio ** 2) + 1) ** 2.5)
            rsens_IP = -(96 * (depth_ratio ** 2) - 6) / (coil_spacing * (4 * (depth_ratio ** 2) + 1) ** (7 / 2))
            csens_QP = 1 - ((2 * depth_ratio) / ((4 * depth_ratio ** 2) + 1) ** 0.5)
            rsens_QP = 2 / ((coil_spacing * (4 * depth_ratio ** 2) + 1) ** (3 / 2))

        return rsens_QP, csens_QP, rsens_IP, csens_IP