# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#	 http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
TensorRT Plugin Loader for MLPerf
Python implementation to load TensorRT plugins similar to trtUtils.h
"""

import os
import ctypes
import ctypes.util
from typing import List, Optional
from pathlib import Path


class PluginLoader:
    """Loads TensorRT plugins for MLPerf evaluation."""
    
    def __init__(self):
        self.plugin_handles = []
        self.loaded_plugins = {}
    
    def load_attention_plugin(self) -> Optional[ctypes.CDLL]:
        """
        Load the Attention plugin.
        
        Returns:
            CDLL object if successful, None otherwise
        """
        plugin_path = os.getenv("ATTENTION_PLUGIN_PATH")
        
        if plugin_path is None:
            # Default to build directory
            build_dir = Path(__file__).parent.parent / "build"
            plugin_path = build_dir / "libAttentionPlugin.so"
            print(f"ATTENTION_PLUGIN_PATH variable is not set. Default to {plugin_path}")
        else:
            print(f"ATTENTION_PLUGIN_PATH: {plugin_path}")
        
        try:
            # Load the plugin library
            plugin_handle = ctypes.CDLL(str(plugin_path))
            self.plugin_handles.append(plugin_handle)
            self.loaded_plugins["attention"] = plugin_handle
            print(f"✓ Successfully loaded Attention plugin from {plugin_path}")
            return plugin_handle
            
        except Exception as e:
            print(f"✗ Cannot open Attention plugin library: {e}")
            return None
    
    def load_int4_gemm_plugin(self) -> Optional[ctypes.CDLL]:
        """
        Load the Int4Gemm plugin.
        
        Returns:
            CDLL object if successful, None otherwise
        """
        plugin_path = os.getenv("INT4_GEMM_PLUGIN_PATH")
        
        if plugin_path is None:
            # Default to build directory
            build_dir = Path(__file__).parent.parent / "build"
            plugin_path = build_dir / "libInt4GemmPlugin.so"
            print(f"INT4_GEMM_PLUGIN_PATH variable is not set. Default to {plugin_path}")
        else:
            print(f"INT4_GEMM_PLUGIN_PATH: {plugin_path}")
        
        try:
            # Load the plugin library
            plugin_handle = ctypes.CDLL(str(plugin_path))
            self.plugin_handles.append(plugin_handle)
            self.loaded_plugins["int4_gemm"] = plugin_handle
            print(f"✓ Successfully loaded Int4Gemm plugin from {plugin_path}")
            return plugin_handle
            
        except Exception as e:
            print(f"✗ Cannot open Int4Gemm plugin library: {e}")
            return None
    
    def load_plugins(self, int4_gemm_plugin: bool = True) -> List[ctypes.CDLL]:
        """
        Load all required plugins.
        
        Args:
            int4_gemm_plugin: Whether to load the Int4Gemm plugin
            
        Returns:
            List of loaded plugin handles
        """
        print("Loading TensorRT plugins...")
        
        # Load Attention plugin
        attention_plugin = self.load_attention_plugin()
        
        # Load Int4Gemm plugin if requested
        int4_gemm_plugin_handle = None
        if int4_gemm_plugin:
            int4_gemm_plugin_handle = self.load_int4_gemm_plugin()
        
        # Return list of successfully loaded plugins
        loaded_handles = []
        if attention_plugin:
            loaded_handles.append(attention_plugin)
        if int4_gemm_plugin_handle:
            loaded_handles.append(int4_gemm_plugin_handle)
        
        print(f"✓ Loaded {len(loaded_handles)} plugins successfully")
        return loaded_handles
    
    def get_plugin_handle(self, plugin_name: str) -> Optional[ctypes.CDLL]:
        """
        Get a specific plugin handle by name.
        
        Args:
            plugin_name: Name of the plugin ("attention" or "int4_gemm")
            
        Returns:
            Plugin handle if loaded, None otherwise
        """
        return self.loaded_plugins.get(plugin_name)
    
    def list_loaded_plugins(self) -> List[str]:
        """
        Get list of loaded plugin names.
        
        Returns:
            List of plugin names
        """
        return list(self.loaded_plugins.keys())
    
    def cleanup(self):
        """Clean up plugin handles."""
        self.plugin_handles.clear()
        self.loaded_plugins.clear()
        print("✓ Plugin handles cleaned up")


# Global plugin loader instance
_plugin_loader = None


def load_plugins(int4_gemm_plugin: bool = True) -> List[ctypes.CDLL]:
    """
    Global function to load plugins.
    
    Args:
        int4_gemm_plugin: Whether to load the Int4Gemm plugin
        
    Returns:
        List of loaded plugin handles
    """
    global _plugin_loader
    
    if _plugin_loader is None:
        _plugin_loader = PluginLoader()
    
    return _plugin_loader.load_plugins(int4_gemm_plugin)


def get_plugin_loader() -> PluginLoader:
    """
    Get the global plugin loader instance.
    
    Returns:
        PluginLoader instance
    """
    global _plugin_loader
    
    if _plugin_loader is None:
        _plugin_loader = PluginLoader()
    
    return _plugin_loader


def cleanup_plugins():
    """Clean up the global plugin loader."""
    global _plugin_loader
    
    if _plugin_loader is not None:
        _plugin_loader.cleanup()
        _plugin_loader = None


# Example usage and testing
if __name__ == "__main__":
    import sys
    
    print("Testing TensorRT Plugin Loader")
    print("=" * 40)
    
    try:
        # Load plugins
        plugin_handles = load_plugins(int4_gemm_plugin=True)
        
        # Get plugin loader instance
        loader = get_plugin_loader()
        
        # List loaded plugins
        loaded_plugins = loader.list_loaded_plugins()
        print(f"\nLoaded plugins: {loaded_plugins}")
        
        # Test getting specific plugin handles
        for plugin_name in loaded_plugins:
            handle = loader.get_plugin_handle(plugin_name)
            if handle:
                print(f"✓ {plugin_name} plugin handle retrieved successfully")
            else:
                print(f"✗ Failed to get {plugin_name} plugin handle")
        
        print("\n✓ Plugin loading test completed successfully!")
        
    except Exception as e:
        print(f"✗ Error during plugin loading test: {e}")
        sys.exit(1)
    finally:
        # Clean up
        cleanup_plugins() 