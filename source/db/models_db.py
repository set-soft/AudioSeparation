# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnología Industrial
# License: GPLv3
# Project: ComfyUI-AudioSeparation
#
# Handles a JSON file with information about the supported models.
# The JSON file is similar to what UVR uses, but with more information.
import json
import logging
import os
from pathlib import Path
from ..utils.misc import NODES_NAME
from ..utils.downloader import download_model as download_model_basic
from ..utils.comfy_notification import send_toast_notification
from .hash_dir import hash_dir
from .hash import is_hash, get_hash

logger = logging.getLogger(f"{NODES_NAME}.models_db")
known_models = None
known_models_mtime = None
# ICON_REMOTE = "\u2601"         # ☁️ Cloud
ICON_REMOTE = "⬇️  "         # "\u2B07"      # ⬇️
ICON_DOWNLOADED = "\U0001F4BE "  # 💾 Floppy Disk
KNOWN_SOURCES = {'Politrees/MDXNet': 'https://huggingface.co/Politrees/UVR_resources/resolve/main/models/MDXNet',
                 'Main/MDX': 'https://huggingface.co/set-soft/audio_separation/resolve/main/MDX',
                 'Main/Demucs': 'https://huggingface.co/set-soft/audio_separation/resolve/main/Demucs', }


def get_db_filename(provided=None):
    if provided is not None:
        return provided
    # Get the directory where this script is located
    script_dir = Path(__file__).resolve().parent

    # Build the path to the JSON file: go up one level, then into models/
    json_path = script_dir / ".." / ".." / "models" / "uvr_model_data.json"
    try:
        return json_path.resolve().relative_to(Path.cwd())
    except ValueError:
        pass
    return json_path.resolve()


def load_known_models(json_path=None):
    """
    Loads the uvr_model_data.json configuration file using a path relative
    to this script's location. This is the most reliable method.
    """
    global known_models
    global known_models_mtime
    json_path = get_db_filename(json_path)

    # Check if we have a fresh db
    if known_models is not None and known_models_mtime == os.path.getmtime(json_path):
        return known_models

    try:
        logger.debug(f"Attempting to load JSON from: {json_path}")

        # Open and load the JSON file
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        known_models = data
        known_models_mtime = os.path.getmtime(json_path)
        return data

    except FileNotFoundError:
        logger.error("Error: The models database was not found at the expected location.")
        logger.error("Please check the directory structure.")
        return None
    except json.JSONDecodeError as e:
        msg = f"Error: The file at '{json_path}' is not a valid JSON file: {e}"
        logger.error(msg)
        raise ValueError(msg)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return None


def save_known_models(data, json_path=None):
    json_path = get_db_filename(json_path)
    backup_path = os.path.join(os.path.dirname(json_path), f".{os.path.basename(json_path)}~")

    try:
        logger.debug(f"Creating DB backup at '{backup_path}'...")
        if os.path.exists(json_path):
            os.rename(json_path, backup_path)

        logger.debug(f"Saving new database to '{json_path}'...")
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=4, sort_keys=True)

        logger.debug("✅ Database updated and saved successfully.")

        global known_models
        global known_models_mtime
        known_models = data
        known_models_mtime = os.path.getmtime(json_path)

    except Exception as e:
        logger.error(f"during database save: {e}")
        logger.info("Attempting to restore from backup...")
        if os.path.exists(backup_path):
            os.rename(backup_path, json_path)
            logger.info("Backup restored.")
        raise


def create_display_name(d, hash, desc, primary_stem, model_t, file_t, downloaded):
    if primary_stem and len(primary_stem) == 1:
        desc = desc.replace(next(iter(primary_stem)), "")
    if model_t and len(model_t) == 1:
        desc = desc.replace(next(iter(model_t)), "")
    if file_t is None or len(file_t) > 1:
        # We can get a name collision i.e. ONNX + safetensors
        file_extra = f" [{d['file_t']}]"
    else:
        file_extra = ""
    return " ".join(desc.split()) + file_extra


def get_models_full(primary_stem=None, model_t=None, file_t=None, json_path=None, downloaded=None, default=None,
                    repeat_dl=False):
    """ Returns a dict with models that satisfy the provided criteria """

    # Allow for multiple values in the filters
    if isinstance(primary_stem, str):
        primary_stem = {primary_stem}
    elif isinstance(primary_stem, list):
        primary_stem = set(primary_stem)
    if isinstance(model_t, str):
        model_t = {model_t}
    if isinstance(file_t, str):
        file_t = {file_t}

    # Filter the models db
    models = load_known_models(json_path)
    found = {}
    found_hashes = {}
    found_disk = {}
    def_sep = []
    on_disk = []
    to_down = []
    on_disk_as_down = []
    for hash, d in models.items():
        # Skip unnamed models, valid, these are models we don't have or support
        try:
            name = d['name']
        except KeyError:
            continue
        # Description is mandatory
        try:
            desc = d['desc']
        except KeyError:
            logger.error(f"Missing `desc` for {name}")
            continue
        # Check the stem
        try:
            if primary_stem is not None:
                mps = d['primary_stem']
                if isinstance(mps, str):
                    if mps not in primary_stem:
                        continue
                else:  # A list
                    if not (set(mps) & primary_stem):
                        continue
        except KeyError:
            logger.error(f"Missing `primary_stem` for {name}")
            continue
        # Check the network type
        try:
            if model_t is not None and d['model_t'] not in model_t:
                continue
        except KeyError:
            logger.error(f"Missing `model_t` for {name}")
            continue
        # Check the container type
        try:
            if file_t is not None and d['file_t'] not in file_t:
                continue
        except KeyError:
            logger.error(f"Missing `file_t` for {name}")
            continue
        filtered_name = create_display_name(d, hash, desc, primary_stem, model_t, file_t, downloaded)
        d['hash'] = hash
        d['filtered_name'] = filtered_name
        if downloaded is not None:
            file_name = downloaded.get(hash)
            d['indicator'] = ICON_DOWNLOADED if file_name is not None else ICON_REMOTE
        else:
            file_name = None
            d['indicator'] = ""
        if default and default == name:
            if downloaded is None:
                def_sep.append(filtered_name)
            else:
                if file_name is not None:
                    def_sep.append(ICON_DOWNLOADED + filtered_name)
                    if repeat_dl:
                        # A copy for ComfyUI, so the user doesn't need to refresh the node and choose the one downloaded
                        # Also helps to allow forcing save a node with the file as "not downloaded"
                        # Is a hack, but is the best I came up
                        on_disk_as_down.append(ICON_REMOTE + filtered_name)
                else:
                    def_sep.append(ICON_REMOTE + filtered_name)
        else:
            if file_name is not None:
                on_disk.append(ICON_DOWNLOADED + filtered_name)
                if repeat_dl:
                    on_disk_as_down.append(ICON_REMOTE + filtered_name)
            else:
                to_down.append(ICON_REMOTE + filtered_name)
        d['hash'] = hash
        found[filtered_name] = d
        found_hashes[hash] = d
        if file_name is not None:
            d['model_path'] = downloaded[hash]
            found_disk[os.path.realpath(file_name)] = d

    return found, found_hashes, found_disk, def_sep + sorted(on_disk) + sorted(to_down) + sorted(on_disk_as_down)


def get_models(primary_stem=None, model_t=None, file_t=None, json_path=None, downloaded=None, default=None, by_hash=False):
    dnames, hashes, fnames, ldnames = get_models_full(primary_stem, model_t, file_t, json_path, downloaded, default)
    return hashes if by_hash else dnames, ldnames


def get_download_url(data):
    try:
        name = data['name']
        dn_t = data['download']
    except KeyError:
        return None
    try:
        return os.path.join(KNOWN_SOURCES[dn_t], name)
    except KeyError:
        logger.error(f"Unknown download source `{dn_t}`")
        return None


def cli_add_db(parser, default_json_file=None):
    default_json_file = default_json_file or get_db_filename()
    parser.add_argument('--json_file', type=str, default=default_json_file,
                        help="Path to the models database JSON file.")


def cli_add_models_and_db(parser):
    # Compute the models dir assuming the script is run from a clone of the repo
    default_json_file = get_db_filename()

    parser.add_argument('--models_dir', type=str, default=os.path.dirname(default_json_file),
                        help="Path to the directory containing model files.")
    cli_add_db(parser, default_json_file=default_json_file)


def download_model(data, models_dir):
    # Check we can download it
    url = get_download_url(data)
    if url is None:
        raise ValueError("Model is not downloadable")

    # Download the file
    name = data['name']
    send_toast_notification(f"Downloading `{name}`", "Download")
    try:
        fname = download_model_basic(url, models_dir, name)
        # Mark it as downloaded
        data['model_path'] = fname
        if data['indicator']:
            data['indicator'] = ICON_DOWNLOADED
        # Notify the user
        send_toast_notification("Finished downloading", "Download", 'success')
        return fname
    except Exception as e:
        raise ValueError(f"Failed to download {name} from {url}\n{e}")


class FilteredModels(object):
    def __init__(self, primary_stem=None, model_t=None, file_t=None, json_path=None, downloaded=None, default=None,
                 repeat_dl=False):
        self.primary_stem = primary_stem
        self.model_t = model_t
        self.file_t = file_t
        self.by_dname, self.by_hash, self.by_fname, self.dnames = get_models_full(primary_stem, model_t, file_t, json_path,
                                                                                  downloaded, default, repeat_dl)

    def get_by_display_name(self, name):
        if name.startswith(ICON_REMOTE) or name.startswith(ICON_DOWNLOADED):
            name = name[name.index(' ')+1:].strip()
        return self.by_dname.get(name)

    def get_by_hash(self, hash):
        return self.by_hash.get(hash)

    def get_by_file_name(self, file):
        return self.by_fname.get(file)

    def get(self, value):
        # If it looks like a hash try it first
        if is_hash(value):
            d = self.by_hash.get(value)
            if d:
                return d
        # Then try by a display name
        d = self.by_dname.get(value)
        if d:
            return d
        # Is this a file name?
        if os.path.isfile(value):
            # Try using its hash
            hash = get_hash(value)
            return self.by_hash.get(hash)
        return None

    def get_display_names(self, clean=False):
        return [m[m.index(' ')+1:].strip() for m in self.dnames] if clean else self.dnames


class ModelsDB(object):
    def __init__(self, models_dir: str, json_path: str = None):
        super().__init__()
        self.models_dir = models_dir
        self.json_path = get_db_filename(json_path)
        self.refresh()

    def refresh(self):
        self.downloaded = hash_dir(self.models_dir)
        self.models = load_known_models(self.json_path)

    def remove(self, data):
        if data is None:
            return
        del self.models[data["hash"]]

    def add(self, hash, data):
        self.models[hash] = data

    def save(self):
        # Remove run-time information
        for k, v in self.models.items():
            try:
                del v["hash"]
            except KeyError:
                pass
            try:
                del v["indicator"]
            except KeyError:
                pass
            try:
                del v["filtered_name"]
            except KeyError:
                pass
            try:
                del v["model_path"]
            except KeyError:
                pass
        save_known_models(self.models, self.json_path)

    def get_filtered(self, primary_stem=None, model_t=None, file_t=None, default=None, repeat_dl=False):
        return FilteredModels(primary_stem=primary_stem, model_t=model_t, file_t=file_t, json_path=self.json_path,
                              downloaded=self.downloaded, default=default, repeat_dl=repeat_dl)
