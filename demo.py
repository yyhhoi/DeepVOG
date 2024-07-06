import deepvog

# Load our pre-trained network
model = deepvog.load_DeepVOG()

focal_length = 12
video_shape = (240, 320)
sensor_size = (3.6, 4.8)
ransac = True

# Initialize the class. It requires information of your camera's focal length and sensor size, which should be available in product manual.
inferer = deepvog.GazeInferer(model, focal_length, video_shape, sensor_size) 

# Fit an eyeball model from "demo.mp4". The model will be stored as the "inferer" instance's attribute.
input_video = 'demo/demo_video_subsampled.mp4'
inferer.process(input_video, mode="Fit", batch_size=2, ransac=ransac)


# After fitting, infer gaze from "demo.mp4" and output the results into "demo_result.csv"
inferer.process(input_video, mode="Infer", 
                output_record_path="demo/demo_output_results.csv", 
                output_video_path="demo/demo_output_video.mp4", batch_size=2)


# You may save the eyeball model to "demo_model.json" for subsequent gaze inference
eye_ball_model_path = "demo/fitted_eyeball_model.json"
inferer.save_eyeball_model(eye_ball_model_path) 

# By loading the eyeball model, you don't need to fit the model again, and can infer the gaze by calling inferer.process(mode='infer') directly.
inferer.load_eyeball_model(eye_ball_model_path) 
