"""
Audio Processing Module using Whisper (Open Source)
"""
import whisper
import librosa
import soundfile as sf
from pathlib import Path
from typing import Dict, Any, Optional
import torch
from loguru import logger


class AudioProcessor:
    """Process audio files and extract transcripts with speaker diarization"""
    
    def __init__(self, model_name: str = "base", device: str = "cpu"):
        """
        Initialize Whisper ASR model
        
        Args:
            model_name: Whisper model size (tiny, base, small, medium, large)
            device: 'cpu' or 'cuda'
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = whisper.load_model(model_name, device=self.device)
        logger.info(f"Loaded Whisper model: {model_name} on {self.device}")
    
    def process_audio(self, audio_path: Path, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Process audio file and return transcript with metadata
        
        Args:
            audio_path: Path to audio file (mp3, wav, etc.)
            metadata: Additional metadata to include
        
        Returns:
            Dict containing transcript and metadata
        """
        try:
            # Load and preprocess audio
            audio_array, sample_rate = librosa.load(
                audio_path, sr=16000, mono=True
            )
            
            # Transcribe using Whisper
            result = self.model.transcribe(
                audio_path,
                fp16=torch.cuda.is_available(),
                language='en',
                task='transcribe',
                verbose=False
            )
            
            # Extract segments with timestamps
            segments = []
            for segment in result['segments']:
                segments.append({
                    'speaker': f"Speaker_{segment.get('speaker', 'Unknown')}",
                    'text': segment['text'].strip(),
                    'start': segment['start'],
                    'end': segment['end'],
                })
            
            return {
                'text': result['text'],
                'segments': segments,
                'language': result.get('language', 'en'),
                'metadata': metadata or {},
                'source_file': str(audio_path),
                'duration': librosa.get_duration(y=audio_array, sr=sample_rate)
            }
            
        except Exception as e:
            logger.error(f"Error processing audio {audio_path}: {e}")
            return {
                'text': '',
                'segments': [],
                'error': str(e),
                'metadata': metadata or {},
                'source_file': str(audio_path)
            }