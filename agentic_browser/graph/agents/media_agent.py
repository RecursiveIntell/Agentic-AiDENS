"""
Media Agent for media file manipulation and conversion.

Handles video/audio conversion, image processing, and transcription.
"""

import json
import subprocess
import shutil
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from .base import BaseAgent
from ..state import AgentState


class MediaAgentNode(BaseAgent):
    """Specialized agent for media processing tasks.
    
    Handles video, audio, and image transformations
    using ffmpeg, imagemagick, and other tools.
    """
    
    AGENT_NAME = "media"
    MAX_STEPS_PER_INVOCATION = 5
    
    SYSTEM_PROMPT = """You are a MEDIA agent. You process video, audio, and image files.

Available actions:
- ffmpeg_convert: { "input": "video.mp4", "output": "video.webm" }
- ffmpeg_extract_audio: { "input": "video.mp4", "output": "audio.mp3" }
- ffmpeg_trim: { "input": "video.mp4", "output": "clip.mp4", "start": "00:01:00", "duration": "30" }
- ffmpeg_info: { "input": "video.mp4" }  # Get media info
- image_resize: { "input": "photo.jpg", "output": "resized.jpg", "width": 800 }
- image_convert: { "input": "image.png", "output": "image.jpg", "quality": 85 }
- image_compress: { "input": "photo.jpg", "output": "compressed.jpg", "quality": 70 }
- done: { "summary": "what was accomplished" }

RULES:
1. Output files go to ~/Downloads or same directory as input
2. Never overwrite original files - always create new output
3. Check input exists before processing
4. For large files (>1GB), warn user about processing time

Respond with JSON:
{
  "action": "ffmpeg_convert|image_resize|...|done",
  "args": { ... },
  "rationale": "brief reason"
}"""

    def __init__(self, config):
        """Initialize media agent."""
        super().__init__(config)
        self._home_dir = Path.home()
    
    @property
    def system_prompt(self) -> str:
        return self.SYSTEM_PROMPT
    
    def execute(self, state: AgentState) -> AgentState:
        """Execute media processing."""
        # Check for required tools
        tools_available = {
            "ffmpeg": shutil.which("ffmpeg") is not None,
            "convert": shutil.which("convert") is not None,  # ImageMagick
        }
        
        task_context = f"""
MEDIA TASK: {state['goal']}

Available tools: ffmpeg={'✓' if tools_available['ffmpeg'] else '✗'}, imagemagick={'✓' if tools_available['convert'] else '✗'}

Files processed:
{chr(10).join(state['files_accessed'][-5:]) or '(none)'}

Data collected:
{json.dumps(state['extracted_data'], indent=2)[:1000]}
"""
        
        messages = self._build_messages(state, task_context)
        
        try:
            response = self.safe_invoke(messages)
            action_data = self._parse_action(response.content)
            
            action = action_data.get("action", "")
            args = action_data.get("args", {})
            
            if action == "done":
                summary = args.get("summary", "Media processing completed")
                return self._update_state(
                    state,
                    messages=[AIMessage(content=response.content)],
                    extracted_data={"media_result": summary},
                )
            
            # Execute the media action
            result = self._execute_action(action, args)
            
            tool_msg = HumanMessage(content=f"Tool '{action}' output:\n{result['message'][:2000]}")
            
            extracted = None
            if result['success'] and result.get('data'):
                extracted = {f"media_{action}": str(result['data'])[:1500]}
            
            return self._update_state(
                state,
                messages=[AIMessage(content=response.content), tool_msg],
                file_accessed=args.get('input'),
                extracted_data=extracted,
                error=result['message'] if not result['success'] else None,
            )
            
        except Exception as e:
            return self._update_state(
                state,
                error=f"Media agent error: {str(e)}",
            )
    
    def _execute_action(self, action: str, args: dict) -> dict:
        """Execute a media action."""
        handlers = {
            "ffmpeg_convert": self._ffmpeg_convert,
            "ffmpeg_extract_audio": self._ffmpeg_extract_audio,
            "ffmpeg_trim": self._ffmpeg_trim,
            "ffmpeg_info": self._ffmpeg_info,
            "image_resize": self._image_resize,
            "image_convert": self._image_convert,
            "image_compress": self._image_compress,
        }
        
        handler = handlers.get(action)
        if not handler:
            return {"success": False, "message": f"Unknown action: {action}"}
        
        try:
            return handler(args)
        except Exception as e:
            return {"success": False, "message": f"Error: {str(e)}"}
    
    def _validate_path(self, path: str, must_exist: bool = True) -> Path:
        """Validate and resolve a path safely."""
        p = Path(path).expanduser().resolve()
        
        # Must be under home directory or /tmp
        if not (str(p).startswith(str(self._home_dir)) or str(p).startswith('/tmp')):
            raise ValueError(f"Path must be under home directory: {path}")
        
        if must_exist and not p.exists():
            raise FileNotFoundError(f"File does not exist: {path}")
        
        return p
    
    def _ffmpeg_convert(self, args: dict) -> dict:
        """Convert media file format."""
        input_path = self._validate_path(args.get("input", ""))
        output_path = self._validate_path(args.get("output", ""), must_exist=False)
        
        cmd = ["ffmpeg", "-y", "-i", str(input_path), str(output_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            return {"success": False, "message": f"FFmpeg error: {result.stderr[:500]}"}
        
        return {
            "success": True,
            "message": f"Converted {input_path.name} → {output_path.name}",
            "data": {"input": str(input_path), "output": str(output_path)}
        }
    
    def _ffmpeg_extract_audio(self, args: dict) -> dict:
        """Extract audio from video."""
        input_path = self._validate_path(args.get("input", ""))
        output_path = self._validate_path(args.get("output", ""), must_exist=False)
        
        cmd = ["ffmpeg", "-y", "-i", str(input_path), "-vn", "-acodec", "libmp3lame", "-q:a", "2", str(output_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            return {"success": False, "message": f"FFmpeg error: {result.stderr[:500]}"}
        
        return {
            "success": True,
            "message": f"Extracted audio to {output_path.name}",
            "data": {"input": str(input_path), "output": str(output_path)}
        }
    
    def _ffmpeg_trim(self, args: dict) -> dict:
        """Trim media file."""
        input_path = self._validate_path(args.get("input", ""))
        output_path = self._validate_path(args.get("output", ""), must_exist=False)
        start = args.get("start", "00:00:00")
        duration = args.get("duration", "30")
        
        cmd = ["ffmpeg", "-y", "-i", str(input_path), "-ss", start, "-t", str(duration), "-c", "copy", str(output_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode != 0:
            return {"success": False, "message": f"FFmpeg error: {result.stderr[:500]}"}
        
        return {
            "success": True,
            "message": f"Trimmed {input_path.name} from {start} for {duration}s",
            "data": {"input": str(input_path), "output": str(output_path)}
        }
    
    def _ffmpeg_info(self, args: dict) -> dict:
        """Get media file info."""
        input_path = self._validate_path(args.get("input", ""))
        
        cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", str(input_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            return {"success": False, "message": f"FFprobe error: {result.stderr[:500]}"}
        
        try:
            info = json.loads(result.stdout)
            duration = info.get("format", {}).get("duration", "unknown")
            size = info.get("format", {}).get("size", "unknown")
            
            return {
                "success": True,
                "message": f"Duration: {duration}s, Size: {int(size)//1024//1024}MB" if size != "unknown" else "Media info retrieved",
                "data": info
            }
        except json.JSONDecodeError:
            return {"success": True, "message": result.stdout[:1000], "data": {}}
    
    def _image_resize(self, args: dict) -> dict:
        """Resize image."""
        input_path = self._validate_path(args.get("input", ""))
        output_path = self._validate_path(args.get("output", ""), must_exist=False)
        width = args.get("width", 800)
        height = args.get("height")
        
        geometry = f"{width}x{height}" if height else f"{width}x"
        
        cmd = ["convert", str(input_path), "-resize", geometry, str(output_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode != 0:
            return {"success": False, "message": f"ImageMagick error: {result.stderr[:500]}"}
        
        return {
            "success": True,
            "message": f"Resized to {geometry}: {output_path.name}",
            "data": {"input": str(input_path), "output": str(output_path)}
        }
    
    def _image_convert(self, args: dict) -> dict:
        """Convert image format."""
        input_path = self._validate_path(args.get("input", ""))
        output_path = self._validate_path(args.get("output", ""), must_exist=False)
        quality = args.get("quality", 85)
        
        cmd = ["convert", str(input_path), "-quality", str(quality), str(output_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode != 0:
            return {"success": False, "message": f"ImageMagick error: {result.stderr[:500]}"}
        
        return {
            "success": True,
            "message": f"Converted {input_path.name} → {output_path.name}",
            "data": {"input": str(input_path), "output": str(output_path)}
        }
    
    def _image_compress(self, args: dict) -> dict:
        """Compress image."""
        input_path = self._validate_path(args.get("input", ""))
        output_path = self._validate_path(args.get("output", ""), must_exist=False)
        quality = args.get("quality", 70)
        
        cmd = ["convert", str(input_path), "-quality", str(quality), str(output_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode != 0:
            return {"success": False, "message": f"ImageMagick error: {result.stderr[:500]}"}
        
        old_size = input_path.stat().st_size // 1024
        new_size = output_path.stat().st_size // 1024 if output_path.exists() else 0
        
        return {
            "success": True,
            "message": f"Compressed {old_size}KB → {new_size}KB ({output_path.name})",
            "data": {"input": str(input_path), "output": str(output_path), "compression": f"{old_size}→{new_size}KB"}
        }
    
    def _parse_action(self, response: str) -> dict:
        """Parse LLM response into action dict."""
        try:
            content = response.strip()
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1])
            
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1:
                content = content[start:end+1]
            
            return json.loads(content)
        except json.JSONDecodeError:
            return {"action": "done", "args": {"summary": "Unable to parse response"}}


def media_agent_node(state: AgentState) -> AgentState:
    """LangGraph node function for media agent."""
    from ..tool_registry import get_tools
    
    tools = get_tools(state.get("session_id", ""))
    if tools:
        agent_config = tools.config
    else:
        from ...config import AgentConfig
        agent_config = AgentConfig(goal=state['goal'])
    
    agent = MediaAgentNode(agent_config)
    return agent.execute(state)
