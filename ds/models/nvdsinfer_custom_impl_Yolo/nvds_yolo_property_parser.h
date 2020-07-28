/**
 * Copyright (c) 2018 2019, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 *
 */

#ifndef __NVDSINFER_YOLO_PROPERTY_PARSER_H__
#define __NVDSINFER_YOLO_PROPERTY_PARSER_H__

#include <glib.h>

#define CONFIG_GROUP_PROPERTY "yolov3"

/** YoloV3 specific parameters. */
#define CONFIG_GROUP_YOLO_KNMS_THRESHOLD "knms-threshold"
#define CONFIG_GROUP_YOLO_KPROB_THRESHOLD "kprob-threshold"

typedef struct
{
  gfloat knms_threshold;
  gfloat kprob_threshold;
  gboolean knms_is_parsed;
  gboolean kprob_is_parsed;
} NvDsParseBboxYoloV3Config;


gboolean yolov3_parse_config_file (NvDsParseBboxYoloV3Config * config,
        gchar * cfg_file_path);

#endif /*__NVDSINFER_YOLO_PROPERTY_PARSER_H__*/

