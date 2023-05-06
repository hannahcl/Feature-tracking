import numpy as np
from scipy.optimize import least_squares
from scipy.interpolate import RectBivariateSpline
import cv2 as cv
from PIL import Image


class LucasKanade:
    def __init__(self, config):

        search_config = config['search']
        self.n_search_stages = search_config['n_stages']
        self.blur_levels = search_config['blur_levels']
        self.search_windows_sz = search_config['search_windows_sz']
        self.evaluation_windows_sz = search_config['evaluation_windows_sz']
        self.use_pyramid = search_config.get('use_pyramid', False)
        self.use_epipolar = search_config.get('use_epipolar', False)
        self.use_prediction = search_config.get('use_prediction', False)
        self.use_inverse_composition = search_config.get('use_inverse_composition', False)
        assert (
            len(self.blur_levels)
            == len(self.search_windows_sz)
            == len(self.evaluation_windows_sz)
            == self.n_search_stages
        )
        
        optimize_config = config['optimize']
        self.method = optimize_config['method']
        self.loss = optimize_config['loss']
        
        run_config = config['run']
        self.n_frames = run_config['n_frames']
        self.n_transformation_params = run_config['n_transformation_params']
        self.n_features = run_config['n_features']
        self.show_and_save_tracks = run_config['show_and_save_tracks']
        self.save_image_index = run_config['save_image_index']  

        self.ground_truth = config['gt']

        
        self.initial_feature_detections: None
        self.tracks: None
        self.all_transformation_params: None

        self.color = np.random.randint(0, 255, (100, 3))

        self.reference_image: None
        self.target_image: None    
        self.target_image_color: None

        self.i_frame = 0


    def find_tracks_in_target_image(self, target_image):

        self.update_target_image(target_image)

        for i_track, track in enumerate(self.tracks[self.i_frame - 1]):

 
            if self.use_prediction:
                initial_guess = self.all_transformation_params[self.i_frame-1][i_track]
            else:
                initial_guess = None


            if self.use_inverse_composition:
                reference_image_warped = self.warp_image(self.all_transformation_params[self.i_frame-1][i_track], self.reference_image)
            else:
                reference_image_warped = self.reference_image

  
            transformation_params_tot = np.zeros(self.n_transformation_params)
            for stage in range(self.n_search_stages):

                window_sz = self.search_windows_sz[stage]
                blur_kernal = self.blur_levels[stage]
                tarnslation_params = transformation_params_tot[0:2]

                search_window_reference = self.resize_windows(
                    reference_image_warped, 
                    track, 
                    window_sz
                )
                search_window_target = self.resize_windows(
                    self.target_image, 
                    track + tarnslation_params, 
                    window_sz
                )

                search_window_reference = cv.blur(
                    search_window_reference, (blur_kernal, blur_kernal)
                )
                search_window_target = cv.blur(
                    search_window_target, (blur_kernal, blur_kernal)
                )

                search_window_center_coordiantes = [
                    np.floor(window_sz / 2),
                    np.floor(window_sz / 2),
                ]

                transformation_params = self.find_min_p(
                    search_window_reference,
                    search_window_target,
                    search_window_center_coordiantes,
                    self.evaluation_windows_sz[stage],
                    initial_guess
                )
                initial_guess = np.zeros(self.n_transformation_params)
                transformation_params_tot += transformation_params

            self.tracks[self.i_frame][i_track] = track + transformation_params_tot
            self.all_transformation_params[self.i_frame][i_track]  = transformation_params_tot

    def find_min_p(self, I1, I2, eval_window_center, window_sz, initial_p=None):

        resfun = lambda p: self.costfunction(I1, I2, eval_window_center, window_sz, p)

        if initial_p is None:
            initial_p = np.zeros(self.n_transformation_params)

        result = least_squares(
            resfun, initial_p, method=self.method, loss=self.loss
        )  
        return result.x

    def costfunction(self, I1, I2, eval_window_center, window_sz, p):
        win_xmin = int(max(
            eval_window_center[0] - np.floor(window_sz / 2), 
            0))
        
        win_xmax = int(min(
            eval_window_center[0] + np.floor(window_sz / 2), 
            I1.shape[0]))

        win_ymin = int(
            max(eval_window_center[1] - np.floor(window_sz / 2), 
                0))
        
        win_ymax = int(
            min(eval_window_center[1] + np.floor(window_sz / 2), 
            I1.shape[1]))

        yy, xx = np.mgrid[win_ymin:win_ymax, win_xmin:win_xmax]

        interp_fn1 = RectBivariateSpline(
            np.arange(I1.shape[0]), np.arange(I1.shape[1]), I1
        )
        interp_fn2 = RectBivariateSpline(
            np.arange(I2.shape[0]), np.arange(I2.shape[1]), I2
        )

        shifted_xx = xx + p[0]
        shifted_yy = yy + p[1]
        initial_window = interp_fn1.ev(yy, xx)
        shifted_window = interp_fn2.ev(shifted_yy, shifted_xx)

        return initial_window.ravel() - shifted_window.ravel()
    
    def get_initial_feature_detections(self, initial_feature_detections):
        if len(initial_feature_detections) < self.n_features:
            self.n_features = len(initial_feature_detections)

        self.tracks = np.zeros(
            (
                self.n_frames,
                self.n_features,
                self.n_transformation_params,
            )
        )

        self.tracks[0] = initial_feature_detections[0:self.n_features]

        self.all_transformation_params = np.zeros(
            (
                self.n_frames,
                self.n_features,
                self.n_transformation_params,
            )
        )

    def resize_windows(self, original_image, center_coordiantes, window_sz):

        win_xmin = int(max(center_coordiantes[0] - np.floor(window_sz / 2), 0))
        win_xmax = int(
            min(
                center_coordiantes[0] + np.floor(window_sz / 2),
                original_image.shape[0] - 1,
            )
        )
        win_ymin = int(max(center_coordiantes[1] - np.floor(window_sz / 2), 0))
        win_ymax = int(
            min(
                center_coordiantes[1] + np.floor(window_sz / 2),
                original_image.shape[1] - 1,
            )
        )

        search_window = original_image[win_ymin:win_ymax, win_xmin:win_xmax]

        return search_window

    def update_target_image(self, target_image):
        self.reference_image = self.target_image
        self.target_image = cv.cvtColor(target_image, cv.COLOR_BGR2GRAY)
        self.target_image_color = target_image
        self.i_frame += 1

    def warp_image(self, transformation_params, image):
        if transformation_params is None:
            return image

        M = np.zeros((2,3))
        M[0,2] = transformation_params[0]
        M[1,2] = transformation_params[1]

        if self.n_transformation_params == 2:
            M[0,0] = 1
            M[0,1] = 0
            M[1,0] = 0
            M[1,1] = 1

        if self.n_transformation_params == 6:
            M[0,0] = transformation_params[3]
            M[0,1] = transformation_params[4]
            M[1,0] = transformation_params[5]
            M[1,1] = transformation_params[6]

        rows, cols = image.shape[:2]

        warped = cv.warpAffine(image, M, (cols, rows))

        return warped


    def show_and_save_tracked_features(self):
        for i, track in enumerate(self.tracks[self.i_frame]):
            self.target_image_color = cv.circle(
                self.target_image_color,
                (int(track[0]), int(track[1])),
                5,
                self.color[i].tolist(),
                -1,
            )

        if self.i_frame in self.save_image_index:

            image_filename = "img" + str(self.i_frame) + ".jpeg"
            img = Image.fromarray(self.target_image_color)
            img.save(image_filename)


        cv.imshow('frame', self.target_image_color)
        k = cv.waitKey(30) & 0xff
        if k == 10:
            return
        

    def compare_tracks_to_ground_truth(self):
        print(self.method, self.loss)

        for i in [3, 10, 50, 100]:
            tracks = self.tracks[i]
            gt = np.array(self.ground_truth[f'tracks_frame_{i}'])
            for j in range(self.n_features):
                error = np.abs(tracks[j] - gt[j])
                print(f"Frame {i} and track {j}")
                print(f"gt: {gt[j]}, estimated: {tracks[j]}, error: {error}")  


            
 

