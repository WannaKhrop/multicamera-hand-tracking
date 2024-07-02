import cv2
import pyrealsense2 as rs
import numpy as np

class camera: 
    def __init__(self, device_name: str, device_id: int):
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # internal data
        self.device_name = device_name
        self.device_id = device_id
        self.pipeline_started=False
        self.intrinsics_saved=False

        # Get device product line for setting a supporting resolution
        #pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        #pipeline_profile = self.config.resolve(pipeline_wrapper)
        #device = pipeline_profile.get_device()
        

        align_to = rs.stream.color
        self.align = rs.align(align_to)

        self.config.enable_device(self.device_id)
        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 1920, 1080, rs.format.rgb8, 30)

        self.intrinsics = rs.intrinsics()

    def take_picture_and_return_color(self):
        #start pipeline
        if not self.pipeline_started:
            self.pipeline.start(self.config)

            #load profile for intrinsics and extrinsics
            active_profile=self.pipeline.get_active_profile()
           
            #get extrinsics
            self.depth_to_color_extrin =  active_profile.get_stream(rs.stream.depth).as_video_stream_profile().get_extrinsics_to( active_profile.get_stream(rs.stream.color))
            self.color_to_depth_extrin =  active_profile.get_stream(rs.stream.color).as_video_stream_profile().get_extrinsics_to( active_profile.get_stream(rs.stream.depth))

            #get depth scale
            self.depth_scale = active_profile.get_device().first_depth_sensor().get_depth_scale()
            #take some pictures till quality is good
            for i in range(1,100):
                self.pipeline.wait_for_frames()
                self.pipeline_started=True

        
        #get frames
        frames = self.pipeline.wait_for_frames()

        #align color and depth stream
        aligned_frames = self.align.process(frames)

        
        self.depth_frame = aligned_frames.get_depth_frame() 
        self.color_frame = aligned_frames.get_color_frame()

        #get intrinsics
        self.depth_intrin = self.depth_frame.get_profile().as_video_stream_profile().get_intrinsics()
        self.color_intrin = self.color_frame.get_profile().as_video_stream_profile().get_intrinsics()

        imgRGB = np.asanyarray(self.color_frame.get_data())
        imgRGB = cv2.cvtColor(imgRGB, cv2.COLOR_BGR2RGB)

        return imgRGB

    def get_depth_data_from_pixel(self, px, py):
        #depth_min = 0.11 #meter
        #depth_max = 1.0 #meter

        #depth_point = rs.rs2_project_color_pixel_to_depth_pixel(self.depth_frame.get_data(), self.depth_scale, depth_min, depth_max, self.depth_intrin, self.color_intrin, self.depth_to_color_extrin, self. color_to_depth_extrin, [px, py])

        #need to check the coordinates of x and y
        try:
            depth=self.depth_frame.get_distance(int(px), int(py))
            dx ,dy, dz = rs.rs2_deproject_pixel_to_point(self.color_intrin, [px,py], depth)
            return [dx, dy, dz]
        except:
            return [-1, -1, -1]

    def __del__(self):
        if self.pipeline_started:
            self.pipeline.stop()

