import deepvog

# Load our pre-trained network
model = deepvog.load_DeepVOG()

focal_length = 12
video_shape = (240, 320)
sensor_size = (3.6, 4.8)

# Initialize the class. It requires information of your camera's focal length and sensor size, which should be available in product manual.
inferer = deepvog.gaze_inferer(model, focal_length, video_shape, sensor_size) 

# Fit an eyeball model from "demo.mp4". The model will be stored as the "inferer" instance's attribute.
input_video = 'test/test_data/testdata_subsampled_video.mp4'
inferer.process(input_video, mode="Fit", batch_size=2, ransac=False)

# After fitting, infer gaze from "demo.mp4" and output the results into "demo_result.csv"
inferer.process(input_video, mode="Infer", output_record_path="test/test_data/demo_results.csv", 
                output_video_path="test/test_data/demo_output_video.mp4", batch_size=2)

# Optional

# You may also save the eyeball model to "demo_model.json" for subsequent gaze inference
inferer.save_eyeball_model("test/test_data/demo_model.json") 

# By loading the eyeball model, you don't need to fit the model again
inferer.load_eyeball_model("test/test_data/demo_model.json") 