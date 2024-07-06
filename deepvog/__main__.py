"""
File: __main__.py
Author: Yuk-Hoi Yiu
Description: 
    The script serves as the entry point of the DeepVOG package. After installing the DeepVOG package, run the command such as: 
    
    $ python -m deepvog --fit ~/video01.mp4 ~/model01.json
    $ python -m deepvog --infer ~/video01.mp4 ~/model01.json ~/model1_video01.csv

    Eyeball model is stored as json. Gaze inference result is stored as a csv file.

    The arguments and further function calls for the entry point are handled below.

"""
import argparse
from argparse import RawTextHelpFormatter
from ast import literal_eval

description_text = """

Belows are the examples of usage. Don't forget to set up camera parameters such as focal length, because it varies from equipment to equipment and is necessaray for accuracy.

    Example #1 - Fitting an eyeball model (using default camera parameters)
    python -m deepvog --fit ~/video01.mp4 ~/model01.json

    Example #2 - Infer gaze (using default camera parameters)
    python -m deepvog --infer ~/video01.mp4 ~/model01.json ~/model1_video01.csv

    Example #3 - Setting up necessary parameters (focal length = 12mm, video shape = (240,320), ... etc)
    python -m deepvog --fit ./vid.mp4 ./model.json -f 12 -vs (240,320) -s (3.6,4.8) -b 32 -g 0

Caution: If your video has an aspect ratio (height/width) other than 0.75, please crop it manually until it reaches the required aspect ratio otherwise the gaze estimate will not be accurate.

    Example #4 - If you cropped the video from (250,390) to (240,360), keep using the argument "-vs 250 390"
    python -m deepvog --fit ./vid.mp4 ./model.json -f 12 -vs (250,390) -s (3.6,4.8) -b 32 -g 0
    
You can also fit and infer a video from a csv table:

    Example #5
    python -m deepvog --table ./table.csv -f 12 -vs (240,320) -s (3.6,4.8) -b 32 -g 0

The csv file must follow the format below:
    
    operation, fit_vid,             infer_vid,              eyeball_model,      result
    fit      , /PATH/fit_vid.mp4,   /PATH/infer_vid.mp4,    /PATH/model.json,   /PATH/output_result.csv
    infer    , /PATH/fit_vid2.mp4,  /PATH/infer_vid2.mp4,   /PATH/model2.json,  /PATH/output_result2.csv
    both     , /PATH/fit_vid3.mp4,  /PATH/infer_vid3.mp4,   /PATH/model3.json,  /PATH/output_result3.csv
    fit      , /PATH/fit_vid4.mp4,  /PATH/infer_vid4.mp4,   /PATH/model4.json,  /PATH/output_result4.csv
    ...
Note: all leading and trailing spaces will be removed.
    
The "operation" column should contain either fit/infer/both: 
    1. fit: it will fit the video specified by the path "fit_vid" and save the model to the path specified by "eyeball_model". Other columns are irrelevant and can be omited.
    2. infer: it will load the model from the path specified in "eyeball_model", and infer the video from the path specified in "infer_vid", and then save the result to the path specified by "result".
    3. both: first performs "fit", then performs "infer".

"""

fit_help = "Fitting an eyeball model. Call with --fit [video_src_path] [eyeball_model_saving_path]."
infer_help = "Inter video from an eyeball model. The eyeball model is fitted using the --fit call. Call with --infer [video_scr_path] [eyeball_model_path] [results_saving_path]"
table_help = "Fit or infer videos from a csv table. The column names of the csv must follow a format (see --help description). Call with --table [csv_path]"
ori_vid_shape_help = "Original and uncropped video shape of your camera output, height and width in pixel. Default = (240, 320)"
flen_help = "Focal length of your camera in mm."
gpu_help = "GPU device number. Default = 0. Parallel processing is not supported."
sensor_help = "Sensor size of your camera digital sensor, height and width in mm. Default = (3.6, 4.8)"
batch_help = "Batch size for forward inference. Default = 512."
visualize_help = "Draw the visualization of ellipse fitting and gaze vector. Call with --visualize [video_output_path]. (Not yet available with --table mode)"
heatmap_help = "Show network's output of segmented pupil heatmap in visualization. Call with --heatmap. (Not yet available with --table mode)"
skip_existed_help = "Available only in table mode. If activated, opetation with existing output paths will be skipped. " 
skip_errors_help = "Available only in table mode. Skip the operation and continue to the next if an error is encountered. Useful if you do not want the job be interrupted by errors."
log_errors_help = "Available only in table mode. Log the encountered errors in a txt file for each table-mode job. Call with --log_errors [output_file]."
no_gaze_help = "If activated, no gaze inference is performed, infer-mode job does not required an eyeball model. Useful if you just want to use pupil segmentation without infering the gaze. "
ransac_help = "Enable RANSAC for robust eyeball fitting. Default = True."

parser = argparse.ArgumentParser(description=description_text, formatter_class=RawTextHelpFormatter)
required = parser.add_argument_group("required arguments")
required.add_argument("--fit", help=fit_help, nargs=2, type=str, metavar=("PATH", "PATH"))
required.add_argument("--infer", help=infer_help, nargs=3, type=str, metavar=("PATH", "PATH", "PATH"))
required.add_argument("--table", help=table_help, type=str, metavar=("PATH"))
parser.add_argument("-f", "--flen", help=flen_help, default=12, type=float, metavar=("FLOAT"))
parser.add_argument("-g", "--gpu", help=gpu_help, default="0", type=str, metavar=("INT"))
parser.add_argument("-vs", "--vidshape", help=ori_vid_shape_help, default="(240, 320)", type=str, metavar=("INT,INT"))
parser.add_argument("-s", "--sensor", help=sensor_help, default="(3.6, 4.8)", type=str, metavar=("FLOAT,FLOAT"))
parser.add_argument("-b", "--batchsize", help=batch_help, default=512, type=int, metavar=("INT"))
parser.add_argument("-v", "--visualize", help=visualize_help, default="", type=str, metavar=("PATH"))
parser.add_argument("-m", "--heatmap", help=heatmap_help, default=False, action="store_true")
parser.add_argument("--skip_existed", help=skip_existed_help, default=False, action="store_true")
parser.add_argument("--skip_errors", help=skip_errors_help, default=False, action="store_true")
parser.add_argument("--log_errors", help=log_errors_help, type=str, default="", metavar=("PATH"))
parser.add_argument("--no_gaze", help=no_gaze_help, default=True, action="store_false")
parser.add_argument("--ransac", help=no_gaze_help, default=True, action="store_true")

args = parser.parse_args()

breakpoint()

# Check there is EXACTLY one argument from --fit, --infer and --table
all_modes_list = [args.fit, args.infer, args.table]
cli_modes_list = all_modes_list[0:3]
num_modes = sum([x is not None for x in all_modes_list])
if num_modes != 1:
    parser.error("Exactly one argument from --fit, --infer and --table is requried")

else:

    # Command line mode
    if sum([x is not None for x in cli_modes_list]) > 0:

        from .jobman import deepvog_jobman_CLI, deepvog_jobman_table_CLI
        flen = args.flen
        gpu = args.gpu
        ori_video_shape = literal_eval(args.vidshape)
        sensor_size = literal_eval(args.sensor)
        batch_size = args.batchsize
        ransac = args.ransac

        # Table mode
        if args.table is not None:
            jobman_table = deepvog_jobman_table_CLI(args.table, gpu, flen,
                                                    ori_video_shape, sensor_size, batch_size,
                                                    skip_errors=args.skip_errors,
                                                    skip_existed=args.skip_existed,
                                                    error_log_path=args.log_errors,
                                                    ransac=ransac)
            jobman_table.run_batch()

        # Fit or Infer mode
        jobman = deepvog_jobman_CLI(gpu, flen, ori_video_shape, sensor_size, batch_size)
        if args.fit is not None:
            vid_src_fitting, eyemodel_save = args.fit
            jobman.fit(vid_path=vid_src_fitting, output_json_path=eyemodel_save, output_video_path=args.visualize,
                       heatmap=args.heatmap, ransac=ransac)
        if args.infer is not None:
            vid_scr_inference, eyemodel_load, result_output = args.infer
            jobman.infer(vid_path=vid_scr_inference,
                         eyeball_model_path=eyemodel_load,
                         output_record_path=result_output,
                         output_video_path=args.visualize,
                         heatmap=args.heatmap,
                         infer_gaze_flag=args.no_gaze)
