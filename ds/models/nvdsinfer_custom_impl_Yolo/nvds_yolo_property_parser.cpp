
#include <string.h>

#include "nvds_yolo_property_parser.h"

#define CHECK_ERROR(error) \
    if (error) { \
        g_printerr ("Error while parsing config file: %s\n", error->message); \
        goto done; \
    }


/* Parse 'yolov3' group. Returns FALSE in case of an error. If any of the
 * properties are in this group are not set through config file, the default values 
 * will be used.
 */
static gboolean
yolov3_parse_props (NvDsParseBboxYoloV3Config * config,
    GKeyFile * key_file, gchar * cfg_file_path)
{
  gboolean ret = FALSE;
  gchar **keys = nullptr;
  gchar **key = nullptr;
  GError *error = nullptr;

  keys = g_key_file_get_keys (key_file, CONFIG_GROUP_PROPERTY, nullptr, &error);
  CHECK_ERROR (error);

  for (key = keys; *key; key++) {
    if (!g_strcmp0 (*key, CONFIG_GROUP_YOLO_KNMS_THRESHOLD)) {
      config->knms_threshold =
          g_key_file_get_double (key_file, CONFIG_GROUP_PROPERTY,
          CONFIG_GROUP_YOLO_KNMS_THRESHOLD, &error);
      config->knms_is_parsed = TRUE;
      CHECK_ERROR (error);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_YOLO_KPROB_THRESHOLD)) {
      config->kprob_threshold =
          g_key_file_get_double (key_file, CONFIG_GROUP_PROPERTY,
          CONFIG_GROUP_YOLO_KPROB_THRESHOLD, &error);
      config->kprob_is_parsed = TRUE;
      CHECK_ERROR (error);
    } else {
      g_printerr ("Unknown key '%s' for group [%s]\n", *key,
          CONFIG_GROUP_PROPERTY);
    }
  }

  ret = TRUE;
done:
  if (error) {
    g_error_free (error);
  }
  if (keys) {
    g_strfreev (keys);
  }
  return ret;
}

/* Parse the yolov3 config file. Returns FALSE in case of an error. */
gboolean
yolov3_parse_config_file (NvDsParseBboxYoloV3Config * config, gchar * cfg_file_path)
{
  GError *error = nullptr;
  gboolean ret = FALSE;
  GKeyFile *cfg_file = g_key_file_new ();

  if (!g_key_file_load_from_file (cfg_file, cfg_file_path, G_KEY_FILE_NONE,
          &error)) {
    g_printerr ("Failed to load config file: %s\n", error->message);
    goto done;
  }

  if (!yolov3_parse_props (config, cfg_file, cfg_file_path)) {
    g_printerr ("Failed to parse group %s\n", CONFIG_GROUP_PROPERTY);
    goto done;
  }
  g_key_file_remove_group (cfg_file, CONFIG_GROUP_PROPERTY, nullptr);

  ret = TRUE;

done:
  if (cfg_file) {
    g_key_file_free (cfg_file);
  }

  if (error) {
    g_error_free (error);
  }
  if (!ret) {
    g_printerr ("** ERROR: <%s:%d>: failed\n", __func__, __LINE__);
  }
  return ret;
}
