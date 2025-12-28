#!/usr/bin/env python3
"""
Configure VS Code settings for local LLM mode.

This script disables VS Code hooks, CAPI-dependent features, telemetry,
and 'phone home' functionality.

Usage:
    python configure_vscode.py [--user | --workspace]
    python configure_vscode.py --dry-run  # Show changes without applying
    python configure_vscode.py --restore  # Restore to default values

Dependencies:
    pip install llm-server  # json-five is included for parsing JSONC with comment preservation
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

from json5 import loads as json5_loads
from json5.loader import ModelLoader
from json5.dumper import dumps as json5_dumps, ModelDumper


# Settings to configure for local LLM mode
LOCAL_LLM_SETTINGS = {
    # ========================================
    # DISABLE CAPI-DEPENDENT FEATURES
    # These features require GitHub/Microsoft backend
    # ========================================
    "github.copilot.chat.codesearch.enabled": False,
    "github.copilot.chat.codesearch.agent.enabled": False,
    "github.copilot.chat.search.semanticTextResults": False,

    # ========================================
    # DISABLE VS CODE CORE TELEMETRY
    # ========================================
    "telemetry.telemetryLevel": "off",

    # Disable update checks
    "update.mode": "none",
    "update.showReleaseNotes": False,

    # Disable online services
    "workbench.enableExperiments": False,
    "workbench.settings.enableNaturalLanguageSearch": False,

    # Disable extension recommendations (fetches from Microsoft)
    "extensions.ignoreRecommendations": True,

    # Disable online help/tips
    "workbench.tips.enabled": False,

    # Disable TypeScript/JavaScript survey prompts
    "typescript.surveys.enabled": False,
    "javascript.surveys.enabled": False,

    # Disable Git autofetch (reduces network calls)
    "git.autofetch": False,

    # Disable remote extension host auto port forwarding
    "remote.autoForwardPorts": False,

    # ========================================
    # ADDITIONAL PRIVACY SETTINGS
    # ========================================
    "settingsSync.keybindingsPerPlatform": False,
    "workbench.editSessions.autoResume": "off",
    "workbench.editSessions.continueOn": "off",

    # ========================================
    # SHELL AND CHAT SETTINGS
    # ========================================
    "application.shellEnvironmentResolutionTimeout": 2,
    "chat.notifyWindowOnResponseReceived": False,
}

# Default values for restore operation
DEFAULT_VALUES = {
    "github.copilot.chat.codesearch.enabled": False,  # Default is false anyway
    "github.copilot.chat.codesearch.agent.enabled": True,
    "github.copilot.chat.search.semanticTextResults": True,
    "telemetry.telemetryLevel": "all",
    "update.mode": "default",
    "update.showReleaseNotes": True,
    "workbench.enableExperiments": True,
    "workbench.settings.enableNaturalLanguageSearch": True,
    "extensions.ignoreRecommendations": False,
    "workbench.tips.enabled": True,
    "typescript.surveys.enabled": True,
    "javascript.surveys.enabled": True,
    "git.autofetch": False,  # Default is false
    "remote.autoForwardPorts": True,
    "settingsSync.keybindingsPerPlatform": True,
    "workbench.editSessions.autoResume": "onReload",
    "workbench.editSessions.continueOn": "prompt",
    "application.shellEnvironmentResolutionTimeout": 10,  # VS Code default
    "chat.notifyWindowOnResponseReceived": True,  # VS Code default
}


def get_vscode_config_path(scope: str = "user", variant: str = "code") -> Path:
    """
    Get the VS Code settings.json path for the given scope and variant.

    Args:
        scope: "user" or "workspace"
        variant: "code", "code-insiders", "codium", or "code-oss"
    """
    if scope == "workspace":
        return Path.cwd() / ".vscode" / "settings.json"

    # Map variant names to directory names
    variant_dirs = {
        "code": "Code",
        "code-insiders": "Code - Insiders",
        "codium": "VSCodium",
        "code-oss": "Code - OSS",
    }
    dir_name = variant_dirs.get(variant, "Code")

    # User settings location varies by platform
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        base = Path(appdata) if appdata else Path.home() / "AppData" / "Roaming"
        return base / dir_name / "User" / "settings.json"
    elif sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / dir_name / "User" / "settings.json"
    else:
        # Linux and others - respect XDG
        config_home = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
        return config_home / dir_name / "User" / "settings.json"


def find_vscode_settings() -> list[tuple[str, Path]]:
    """Find all VS Code variant settings files that exist."""
    found = []
    for variant in ["code", "code-insiders", "codium", "code-oss"]:
        path = get_vscode_config_path("user", variant)
        if path.exists():
            found.append((variant, path))
    return found


def load_settings(path: Path) -> tuple[Dict[str, Any], Any]:
    """
    Load settings from JSON file, handling JSONC (JSON with comments).

    Returns a tuple of (settings_dict, model_or_none).
    The model is needed for comment-preserving saves with json-five.
    """
    if not path.exists():
        return {}, None

    content = path.read_text(encoding="utf-8")

    if not content.strip():
        return {}, None

    try:
        # Load as dict for manipulation
        settings_dict = json5_loads(content)
        # Also load as model for round-trip comment preservation
        model = json5_loads(content, loader=ModelLoader())
        return settings_dict, model
    except Exception as e:
        print(f"Warning: Could not parse {path}: {e}")
        return {}, None


def backup_settings(path: Path) -> Path | None:
    """Create a backup of settings file. Returns backup path or None if no backup needed."""
    if not path.exists():
        return None

    # Create backup with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = path.with_suffix(f".backup_{timestamp}.json")

    try:
        import shutil
        shutil.copy2(path, backup_path)
        print(f"Backup created: {backup_path}")
        return backup_path
    except Exception as e:
        print(f"Warning: Could not create backup: {e}")
        return None


def update_model_with_settings(model: Any, settings: Dict[str, Any]) -> None:
    """
    Update a json-five model with new settings values in-place.
    Preserves existing comments and formatting.
    """
    from json5.model import KeyValuePair
    from json5.dumper import modelize

    if not hasattr(model, 'value') or not hasattr(model.value, 'key_value_pairs'):
        return

    # Build a map of existing keys to their index in the list
    # Note: key.characters includes quotes, so we strip them for comparison
    key_value_pairs = model.value.key_value_pairs
    existing_keys = {}
    for idx, kvp in enumerate(key_value_pairs):
        if hasattr(kvp.key, 'characters'):
            # Strip surrounding quotes from the key string
            key_str = kvp.key.characters.strip('"\'')
        else:
            key_str = str(kvp.key).strip('"\'')
        existing_keys[key_str] = idx

    # Update existing keys and track new ones
    for key, value in settings.items():
        if key in existing_keys:
            # KeyValuePair is a NamedTuple (immutable), so create a new one
            idx = existing_keys[key]
            old_kvp = key_value_pairs[idx]
            new_kvp = KeyValuePair(
                key=old_kvp.key,  # Preserve original key formatting
                value=modelize(value),
            )
            key_value_pairs[idx] = new_kvp
        else:
            # Add new key-value pair at the end
            new_kvp = KeyValuePair(
                key=modelize(key),
                value=modelize(value),
            )
            key_value_pairs.append(new_kvp)


def save_settings(path: Path, settings: Dict[str, Any], model: Any = None) -> bool:
    """
    Save settings to JSON file with proper formatting.
    If model is provided, preserves comments using json-five.
    Returns True on success.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            if model is not None:
                # Update the model with new settings and dump with comment preservation
                update_model_with_settings(model, settings)
                f.write(json5_dumps(model, dumper=ModelDumper()))
            else:
                # No model, just dump as formatted JSON5
                f.write(json5_dumps(settings, indent=4))
        print(f"Settings saved to: {path}")
        return True
    except PermissionError:
        print(f"Error: Permission denied writing to {path}")
        print("Try running with elevated privileges or check file permissions.")
        return False
    except Exception as e:
        print(f"Error saving settings: {e}")
        return False


def apply_settings(
    current: Dict[str, Any],
    new_settings: Dict[str, Any],
    dry_run: bool = False
) -> tuple[Dict[str, Any], bool]:
    """Apply new settings and return (updated settings dict, has_changes)."""
    changes = []

    for key, value in new_settings.items():
        old_value = current.get(key)
        if old_value != value:
            changes.append((key, old_value, value))
            if not dry_run:
                current[key] = value

    if changes:
        print("\nSettings changes:")
        for key, old, new in changes:
            old_str = json.dumps(old) if old is not None else "(not set)"
            new_str = json.dumps(new)
            print(f"  {key}: {old_str} -> {new_str}")
    else:
        print("\nNo changes needed - settings already configured.")

    return current, bool(changes)


def main():
    parser = argparse.ArgumentParser(
        description="Configure VS Code settings for local LLM mode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                  # Apply to user settings
  %(prog)s --workspace      # Apply to workspace settings
  %(prog)s --dry-run        # Show what would change
  %(prog)s --restore        # Restore default values
  %(prog)s --variant codium # Apply to VSCodium
  %(prog)s --path /path/to/settings.json  # Custom path
  %(prog)s --find           # Find all VS Code installations
"""
    )
    parser.add_argument(
        '--workspace', action='store_true',
        help='Apply to workspace settings (.vscode/settings.json)'
    )
    parser.add_argument(
        '--variant', choices=['code', 'code-insiders', 'codium', 'code-oss'],
        default='code',
        help='VS Code variant (default: code)'
    )
    parser.add_argument(
        '--path', type=Path,
        help='Custom path to settings.json'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Show what would change without applying'
    )
    parser.add_argument(
        '--restore', action='store_true',
        help='Restore settings to default values'
    )
    parser.add_argument(
        '--list', action='store_true',
        help='List all settings that would be modified'
    )
    parser.add_argument(
        '--find', action='store_true',
        help='Find all VS Code installations with settings'
    )
    parser.add_argument(
        '--all', action='store_true',
        help='Apply to all found VS Code variants'
    )
    parser.add_argument(
        '--no-backup', action='store_true',
        help='Skip creating backup of existing settings'
    )

    args = parser.parse_args()

    # Handle info-only commands first (no json-five needed)
    if args.list:
        print("Settings configured for local LLM mode:\n")
        for key, value in LOCAL_LLM_SETTINGS.items():
            print(f"  {key}: {json.dumps(value)}")
        return

    if args.find:
        found = find_vscode_settings()
        if found:
            print("Found VS Code settings files:\n")
            for variant, path in found:
                print(f"  {variant}: {path}")
        else:
            print("No VS Code settings files found.")
            print("\nExpected locations:")
            for variant in ["code", "code-insiders", "codium", "code-oss"]:
                print(f"  {variant}: {get_vscode_config_path('user', variant)}")
        return

    # Handle --all flag
    if args.all:
        found = find_vscode_settings()
        if not found:
            print("No VS Code installations found.")
            return
        for variant, path in found:
            print(f"\n{'='*60}")
            print(f"Configuring {variant}: {path}")
            print('='*60)
            configure_settings_file(path, args)
        return

    # Determine settings path
    if args.path:
        settings_path = args.path
    elif args.workspace:
        settings_path = get_vscode_config_path("workspace")
    else:
        settings_path = get_vscode_config_path("user", args.variant)

    configure_settings_file(settings_path, args)


def configure_settings_file(settings_path: Path, args) -> bool:
    """Configure a single settings file. Returns True on success."""
    print(f"VS Code settings file: {settings_path}")

    # Load current settings (returns tuple with model for comment preservation)
    current_settings, model = load_settings(settings_path)

    # Determine which settings to apply
    if args.restore:
        print("\nRestoring default values...")
        new_settings = DEFAULT_VALUES
    else:
        print("\nApplying local LLM settings...")
        new_settings = LOCAL_LLM_SETTINGS

    # Apply settings
    updated, has_changes = apply_settings(current_settings, new_settings, dry_run=args.dry_run)

    if not args.dry_run and has_changes:
        # Create backup unless disabled
        if not args.no_backup and settings_path.exists():
            if backup_settings(settings_path) is None:
                print("Aborting due to backup failure.")
                return False

        if save_settings(settings_path, updated, model):
            print("\nDone! Restart VS Code for changes to take effect.")
            return True
        else:
            return False
    elif args.dry_run:
        print("\n(Dry run - no changes made)")
        return True
    else:
        # No changes needed
        return True


if __name__ == "__main__":
    main()
