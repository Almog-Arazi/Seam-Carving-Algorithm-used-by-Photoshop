import numpy as np
from PIL import Image
from numba import jit
from tqdm import tqdm
from abc import abstractmethod, abstractstaticmethod
from os.path import basename
from typing import List
import functools

    
def NI_decor(fn):
    def wrap_fn(self, *args, **kwargs):
        try:
            return fn(self, *args, **kwargs)
        except NotImplementedError as e:
            print(e)
    return wrap_fn


class SeamImage:
    def __init__(self, img_path: str, vis_seams: bool=True):
        """ SeamImage initialization.

        Parameters:
            img_path (str): image local path
            vis_seams (bool): if true, another version of the original image shall be store, and removed seams should be marked on it
        """
        #################
        # Do not change #
        #################
        self.path = img_path
        
        self.gs_weights = np.array([[0.299, 0.587, 0.114]]).T
        
        self.rgb = self.load_image(img_path)
        self.resized_rgb = self.rgb.copy()

        self.vis_seams = vis_seams
        if vis_seams:
            self.seams_rgb = self.rgb.copy()
        
        self.h, self.w = self.rgb.shape[:2]
        
        try:
            self.gs = self.rgb_to_grayscale(self.rgb)
            self.resized_gs = self.gs.copy()
            self.cumm_mask = np.ones_like(self.gs, dtype=bool).squeeze()
        except NotImplementedError as e:
            print(e)

        try:
            self.E = self.calc_gradient_magnitude()
        except NotImplementedError as e:
            print(e)
        #################

        # additional attributes you might find useful
        self.seam_history = []
        self.seam_balance = 0

        # This might serve you to keep tracking original pixel indices 
        self.idx_map_h, self.idx_map_v = np.meshgrid(range(self.w), range(self.h))

    @NI_decor
    def rgb_to_grayscale(self, np_img):
        """ Converts a np RGB image into grayscale (using self.gs_weights).
        Parameters
            np_img : ndarray (float32) of shape (h, w, 3) 
        Returns:
            grayscale image (float32) of shape (h, w, 1)

        Guidelines & hints:
            Use NumpyPy vectorized matrix multiplication for high performance.
            To prevent outlier values in the boundaries, we recommend to pad them with 0.5
        """
        np_img = np_img.astype(np.float32)
        grayscale = np.dot(np_img, self.gs_weights)

        # padding
        grayscale[0, :] = 0.5
        grayscale[-1, :] = 0.5
        grayscale[:, 0] = 0.5
        grayscale[:, -1] = 0.5
        
        return grayscale
      

    @NI_decor
    def calc_gradient_magnitude(self):
        """ Calculate gradient magnitude of a grayscale image

        Returns:
            A gradient magnitude image (float32) of shape (h, w)

        Guidelines & hints:
            - In order to calculate a gradient of a pixel, only its neighborhood is required.
            - keep in mind that values must be in range [0,1]
            - np.gradient or other off-the-shelf tools are NOT allowed, however feel free to compare yourself to them
        """
        gs = self.resized_gs.squeeze()
        # Calculate horizontal and vertical differences using np.roll
        diff_horizontal = gs - np.roll(gs, shift=-1, axis=1)
        diff_vertical   = gs - np.roll(gs, shift=-1, axis=0)
        
        # Compute the gradient magnitude using np.hypot, which computes sqrt(diff_horizontal**2 + diff_vertical**2)
        gradient = np.hypot(diff_horizontal, diff_vertical)
        
        return gradient


    def update_ref_mat(self):
        for i, s in enumerate(self.seam_history[-1]):
            self.idx_map_h[i, s:] += 1

    def reinit(self):
        """
        Re-initiates instance and resets all variables.
        """
        self.__init__(img_path=self.path)

    @NI_decor
    def calc_C(self):
        """ Calculates the matrices C_L, C_V, C_R (forward-looking cost) for the M matrix
        Returns:
            C_L, C_V, C_R matrices (float32) of shape (h, w)
        """
        # Squeeze the greyscale image from (h, w, 1) to (h, w)
        gs_img = self.resized_gs.squeeze()

        # Calculate the cost of the new edges (forward-looking cost)
        middle = np.roll(gs_img, 1, axis=0)
        left = np.roll(gs_img, 1, axis=1)
        right = np.roll(gs_img, -1, axis=1)

        c_v = np.abs(right - left)
        c_l = c_v + np.abs(middle - left)
        c_r = c_v + np.abs(middle - right)
        # Remove a pixel from the first column - no up-left pixel
        c_l[:, 0] = np.inf
        # Remove a pixel from the last column - no up-right pixel
        c_r[:, -1] = np.inf

        return c_l, c_v, c_r

    @staticmethod
    def load_image(img_path, format='RGB'):
        return np.asarray(Image.open(img_path).convert(format)).astype('float32') / 255.0

    def paint_seams(self):
        # paint seams in red
        cumm_mask_rgb = np.stack([self.cumm_mask] * 3, axis=2)
        self.seams_rgb = np.where(cumm_mask_rgb, self.seams_rgb, [1,0,0])

    def update_cumm_mask(self):
        # Updates seam in cumm_mask
        s = self.seam_history[-1]
        for i, s_i in enumerate(s):
            self.cumm_mask[self.idx_map_v[i,s_i], self.idx_map_h[i,s_i]] = False


    def seams_removal(self, num_remove: int):
        """ Iterates num_remove times and removes num_remove vertical seams

        Parameters:
            num_remove (int): number of vertical seams to be removed

        Guidelines & hints:
        As taught, the energy is calculated from top to bottom.
        You might find the function np.roll useful.

        This step can be divided into a couple of steps:
            i) init/update matrices (E, mask) where:
                - E is the gradient magnitude matrix
                - mask is a boolean matrix for removed seams
            iii) find the best seam to remove and store it
            iv) index update: when a seam is removed, index mapping should be updated in order to keep track indices for next iterations
            v) seam removal: create the carved image with the chosen seam (and update seam visualization if desired)
            Note: the flow described below is a recommendation. You may implement seams_removal as you wish, but it needs to support:
            - removing seams a couple of times (call the function more than once)
            - visualize the original image with removed seams marked in red (for comparison)
        """
        for _ in tqdm(range(num_remove)):
            self.E = self.calc_gradient_magnitude()
            self.mask = np.ones_like(self.E, dtype=bool)

            seam = self.find_minimal_seam()
            self.seam_history.append(seam)
            if self.vis_seams:
                self.update_cumm_mask()
                self.update_ref_mat()
            self.remove_seam(seam)

        if self.vis_seams:
            self.paint_seams()
        self.seam_history = []

    @NI_decor
    def find_minimal_seam(self) -> List[int]:
        """
        Finds the seam with the minimal energy.
        Returns:
            The found seam, represented as a list of indexes
        """
        raise NotImplementedError("TODO: Implement SeamImage.find_minimal_seam in one of the subclasses")


    @NI_decor
    def remove_seam(self, seam: List[int]):
        """ Removes a seam from self.rgb (you may create a resized version, like self.resized_rgb)

        Guidelines & hints:
        In order to apply the removal, you might want to extend the seam mask to support 3 channels (rgb) using:
        3d_mak = np.stack([1d_mask] * 3, axis=2)
        ...and then use it to create a resized version.

        :arg seam: The seam to remove
        """
        
        #update mask mat to sign the seam
        rows = np.arange(len(seam))
        self.mask[rows, seam] = False

        d3_mak = np.stack([self.mask] * 3, axis=2)

        # Removes a seam from self.resized_rgb and and self.resized_gs using mask
        self.resized_gs = self.resized_gs[self.mask].reshape(self.h, self.w-1)
        self.resized_rgb = self.resized_rgb[d3_mak].reshape(self.h, self.w - 1, 3)
        self.h, self.w = self.resized_gs.shape[0], self.resized_gs.shape[1]


    @NI_decor
    def rotate_mats(self, clockwise: bool):
        """
        Rotates the matrices either clockwise or counter-clockwise.
        """
        k = -1 if clockwise else 1
        # rotate images
        self.rgb = np.rot90(self.rgb, k, axes=(0, 1))
        self.gs = np.rot90(self.gs, k, axes=(0, 1))
        self.resized_rgb = np.rot90(self.resized_rgb, k, axes=(0, 1))
        self.resized_gs = np.rot90(self.resized_gs, k, axes=(0, 1))
        self.E = np.rot90(self.E, k, axes=(0, 1))

        # if vis(rotate seams visual)
        if self.vis_seams:
            self.cumm_mask = np.rot90(self.cumm_mask, k, axes=(0, 1))
            self.seams_rgb = np.rot90(self.seams_rgb, k, axes=(0, 1))

        # rotate indexs matrix (rotate different side for v)
        self.idx_map_h = np.rot90(self.idx_map_h, k, axes=(0, 1))
        self.idx_map_v = np.rot90(self.idx_map_v, -k, axes=(0, 1))
        
        # update hight and width
        self.h, self.w = self.resized_gs.shape[0], self.resized_gs.shape[1]

        self.idx_map_h, self.idx_map_v = self.idx_map_v, self.idx_map_h

    @NI_decor
    def seams_removal_vertical(self, num_remove: int):
        """ A wrapper for removing num_remove horizontal seams (just a recommendation)

        Parameters:
            num_remove (int): umber of vertical seam to be removed
        """
        self.seams_removal(num_remove)

    @NI_decor
    def seams_removal_horizontal(self, num_remove: int):
        """ Removes num_remove horizontal seams by rotating the image, removing vertical seams, and restoring the original rotation.

        Parameters:
            num_remove (int): number of horizontal seam to be removed
        """
        #  Rotate the image so horizontal seams become vertical.
        self.rotate_mats(clockwise=True)
        # Remove vertical seams (which correspond to the original horizontal seams).
        self.seams_removal(num_remove)
        # Rotate the image back to its original orientation.
        self.rotate_mats(clockwise=False)

    """
    BONUS SECTION
    """

    @NI_decor
    def seams_addition(self, num_add: int):
        """ BONUS: adds num_add seams to the image

            Parameters:
                num_add (int): number of horizontal seam to be removed

            Guidelines & hints:
            - This method should be similar to removal
            - You may use the wrapper functions below (to support both vertical and horizontal addition of seams)
            - Visualization: paint the added seams in green (0,255,0)

        """
        raise NotImplementedError("TODO: ")




    @NI_decor
    def seams_addition_horizontal(self, num_add: int):
        """ A wrapper for removing num_add horizontal seams (just a recommendation)

        Parameters:
            num_add (int): number of horizontal seam to be added

        Guidelines & hints:
            You may find np.rot90 function useful

        """
        self.rotate_mats(clockwise=True)
        self.seams_addition(num_add)
        self.rotate_mats(clockwise=False)
        

    @NI_decor
    def seams_addition_vertical(self, num_add: int):
        """ A wrapper for removing num_add vertical seams (just a recommendation)

        Parameters:
            num_add (int): number of vertical seam to be added
        """

        self.seams_addition(num_add)

class GreedySeamImage(SeamImage):
    """Implementation of the Seam Carving algorithm using a greedy approach"""
    @NI_decor
    def find_minimal_seam(self) -> List[int]:
        """
        Finds the minimal seam by using a greedy algorithm.

        Guidelines & hints:
        The first pixel of the seam should be the pixel with the lowest cost.
        Every row chooses the next pixel based on which neighbor has the lowest cost.
        """
        seam = []
        h, w = self.E.shape
        j = int(np.argmin(self.E[0]))
        seam.append(j)

        c_l, c_v, c_r = self.calc_C()

        for i in range(1,h):
            candidates = []
            for nj in (j-1, j, j+1):
                if 0 <= nj < w:
                    pixe_energy = self.E[i, nj]
                    #pick the right C formula
                    if nj < j: # Left
                        c = c_r[i, nj]
                    elif nj == j: # Vertical
                        c = c_v[i, nj]
                    else: # Right
                        c = c_l[i, nj]
                    candidates.append((pixe_energy+c, nj))
            _, j = min(candidates)
            seam.append(j)
        return seam


class DPSeamImage(SeamImage):
    """
    Implementation of the Seam Carving algorithm using dynamic programming (DP).
    """
    def __init__(self, *args, **kwargs):
        """ DPSeamImage initialization.
        """
        super().__init__(*args, **kwargs)
        try:
            self.backtrack_mat = np.zeros((self.h, self.w, 2), dtype=np.int32)
            self.M = self.calc_M()
        except NotImplementedError as e:
            print(e)

    @NI_decor
    def find_minimal_seam(self) -> List[int]:
        """
        Finds the minimal seam by using dynamic programming.

        Guidelines & hints:
        As taught, the energy is calculated from top to bottom.
        You might find the function np.roll useful.

        This step can be divided into a couple of steps:
            i) init/update matrices (M, backtracking matrix) where:
                - M is the cost matrix
                - backtracking matrix is an idx matrix used to track the minimum seam from bottom up
            ii) fill in the backtrack matrix corresponding to M
            iii) seam backtracking: calculates the actual indices of the seam
        """
        # build/update M and backtrack_mat
        self.M = self.calc_M()  
        h, w = self.M.shape

        #  backtrack from the bottom
        seam = np.zeros(h, dtype=np.int32)
        #  start at the minimal-cost pixel in the last row
        seam[-1] = int(np.argmin(self.M[-1]))
        #  for each row above, follow the pointer in backtrack_mat
        for i in range(h - 1, 0, -1):
            # backtrack_mat[i, j] == [i-1, prev_col]
            _, prev_col = self.backtrack_mat[i, seam[i]]
            seam[i - 1] = prev_col

        #  return as a Python list
        return seam.tolist()

    @NI_decor
    def calc_M(self):
        """ Calculates the matrix M discussed in lecture (with forward-looking cost)

        Returns:
            An energy matrix M (float32) of shape (h, w)

        Guidelines & hints:
            As taught, the energy is calculated from top to bottom.
            You might find the function 'np.roll' useful.
        """
        E = np.squeeze(self.E)
        h, w = E.shape

        # Start by copying E into M
        M = E.copy()
        self.helper_mat = np.zeros((h, w, 2), dtype=np.int32)
        
        cV, cL, cR = self.calc_C()
        
        for i in range(1, h):
            prev_M = M[i - 1]
            
            left_candidate = np.roll(prev_M, 1)
            left_candidate[0] = np.inf
            left_candidate = left_candidate + cL[i]

            center_candidate = prev_M + cV[i]

            right_candidate = np.roll(prev_M, -1)
            right_candidate[-1] = np.inf
            right_candidate = right_candidate + cR[i]
            
            candidates = np.vstack((left_candidate, center_candidate, right_candidate))
            min_costs = np.min(candidates, axis=0)
            argmins = np.argmin(candidates, axis=0)
            
            M[i] = E[i] + min_costs
            
            shifts = np.where(argmins == 0, -1, np.where(argmins == 1, 0, 1))
            prev_indices = np.arange(w) + shifts
            prev_indices = np.clip(prev_indices, 0, w - 1)
            self.helper_mat[i] = np.column_stack((np.full(w, i - 1), prev_indices))
        
        self.backtrack_mat = self.helper_mat.copy()
        return M

    def init_mats(self):
        self.M = self.calc_M()
        self.backtrack_mat = np.zeros((self.h, self.w, 2), dtype=np.int32)

    @staticmethod
    @jit(nopython=True)
    def calc_bt_mat(M, E, GS, backtrack_mat):
        """ Fills the BT back-tracking index matrix. This function is static in order to support Numba. To use it, uncomment the decorator above.

        Recommended parameters (member of the class, to be filled):
            M: np.ndarray (float32) of shape (h,w)
            E: np.ndarray (float32) of shape (h,w)
            GS: np.ndarray (float32) of shape (h,w)
            backtrack_mat: np.ndarray (int32) of shape (h,w): to be filled here

        Guidelines & hints:
            np.ndarray is a reference type. Changing it here may affect it on the outside.
        """
        raise NotImplementedError("TODO: Implement DPSeamImage.calc_bt_mat")
       


def scale_to_shape(orig_shape: np.ndarray, scale_factors: list):
    """ Converts scale into shape

    Parameters:
        orig_shape (np.ndarray): original shape [y,x]
        scale_factors (list): scale factors for y,x respectively

    Returns
        the new shape
    """
    new_shape = np.array(orig_shape) * np.array(scale_factors)
    return np.round(new_shape).astype(int)


def resize_seam_carving(seam_img: SeamImage, shapes: np.ndarray):
    """ Resizes an image using Seam Carving algorithm

    Parameters:
        seam_img (SeamImage) The SeamImage instance to resize
        shapes (np.ndarray): desired shape (y,x)

    Returns
        the resized rgb image
    """
    # Reset the seam image to its original state.
    seam_img.reinit()
    
    # Unpack original and target dimensions.
    (orig_height, orig_width), (target_height, target_width) = shapes
    
    # Determine the number of seams to remove in each dimension.
    vertical_seams = abs(orig_width - target_width)
    horizontal_seams = abs(orig_height - target_height)
    
    # Remove vertical seams for width adjustment.
    seam_img.seams_removal_vertical(vertical_seams)
    # Remove horizontal seams for height adjustment.
    seam_img.seams_removal_horizontal(horizontal_seams)
    
    return seam_img.resized_rgb



def bilinear(image, new_shape):
    """
    Resizes an image to new shape using bilinear interpolation method
    :param image: The original image
    :param new_shape: a (height, width) tuple which is the new shape
    :returns: the image resized to new_shape
    """
    in_height, in_width, _ = image.shape
    out_height, out_width = new_shape
    new_image = np.zeros(new_shape)
    ###Your code here###
    def get_scaled_param(org, size_in, size_out):
        scaled_org = (org * size_in) / size_out
        scaled_org = min(scaled_org, size_in - 1)
        return scaled_org
    scaled_x_grid = [get_scaled_param(x,in_width,out_width) for x in range(out_width)]
    scaled_y_grid = [get_scaled_param(y,in_height,out_height) for y in range(out_height)]
    x1s = np.array(scaled_x_grid, dtype=int)
    y1s = np.array(scaled_y_grid,dtype=int)
    x2s = np.array(scaled_x_grid, dtype=int) + 1
    x2s[x2s > in_width - 1] = in_width - 1
    y2s = np.array(scaled_y_grid,dtype=int) + 1
    y2s[y2s > in_height - 1] = in_height - 1
    dx = np.reshape(scaled_x_grid - x1s, (out_width, 1))
    dy = np.reshape(scaled_y_grid - y1s, (out_height, 1))
    c1 = np.reshape(image[y1s][:,x1s] * dx + (1 - dx) * image[y1s][:,x2s], (out_width, out_height, 3))
    c2 = np.reshape(image[y2s][:,x1s] * dx + (1 - dx) * image[y2s][:,x2s], (out_width, out_height, 3))
    new_image = np.reshape(c1 * dy + (1 - dy) * c2, (out_height, out_width, 3)).astype(int)
    return new_image


