import ICP
icp = ICP.ICP(
            #  model_image = "IMG_7094.JPG",
            #  data_image = "screen_1920x1080_4.png",
            model_image = "maskout.bmp",
             data_image = "maskout_-15.bmp",
             binary_or_color = "binary",
             iterations = 20,
             auto_select_model_and_data = 1,
             connectivity_threshold = 8,
             calculation_image_size = 200,
            display_step = 1,
            debug = 1 )
icp.icp()
icp.display_results()
icp.display_results_as_movie()