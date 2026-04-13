import os
import subprocess
import platform
import shutil
from pathlib import Path
import tempfile
import json
from dataclasses import dataclass
from typing import Optional, Union, List, Dict
import logging
import re
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class WhisperInstallation:
    base_path: Path
    executable_path: Path
    model_path: Path

class VisemeProcessor:
    WHISPER_REPO = "https://github.com/ggerganov/whisper.cpp.git"
    DEFAULT_MODEL = "base.en"
    
    def __init__(self, installation_path: Optional[Union[str, Path]] = None):
        """
        Initialize the VisemeProcessor.
        
        Args:
            installation_path: Optional path where whisper.cpp should be installed.
                             If None, uses a user-specific directory.
        """
        if installation_path is None:
            installation_path = Path.home() / '.viseme_processor'
        self.installation_path = Path(installation_path)
        self.whisper_path = self.installation_path / 'whisper.cpp'
        self._whisper_installation = None

    def _check_dependencies(self) -> bool:
        """Check if required system dependencies are installed."""
        dependencies = ['git', 'make', 'gcc']
        missing = []
        
        for dep in dependencies:
            if shutil.which(dep) is None:
                missing.append(dep)
        
        if missing:
            raise RuntimeError(f"Missing required dependencies: {', '.join(missing)}")
        
        return True

    def _clone_whisper(self) -> None:
        """Clone the whisper.cpp repository."""
        if not self.whisper_path.exists():
            logger.info("Cloning whisper.cpp repository...")
            subprocess.run(['git', 'clone', self.WHISPER_REPO, str(self.whisper_path)], 
                         check=True)
        else:
            logger.info("whisper.cpp repository already exists")

    def _build_whisper(self) -> None:
        """Build the whisper.cpp project."""
        logger.info("Building whisper.cpp...")
        subprocess.run(['make', '-j'], cwd=str(self.whisper_path), check=True)

    def _download_model(self, model_name: str = DEFAULT_MODEL) -> Path:
        """Download the specified whisper model."""
        model_path = self.whisper_path / 'models' / f'ggml-{model_name}.bin'
        
        if not model_path.exists():
            logger.info(f"Downloading {model_name} model...")
            script_path = self.whisper_path / 'models' / 'download-ggml-model.sh'
            subprocess.run(['sh', str(script_path), model_name], 
                         cwd=str(self.whisper_path), check=True)
        else:
            logger.info(f"Model {model_name} already exists")
            
        return model_path

    def ensure_installed(self) -> WhisperInstallation:
        """Ensure whisper.cpp is installed and return installation info."""
        if self._whisper_installation is not None:
            return self._whisper_installation
            
        self._check_dependencies()
        self._clone_whisper()
        self._build_whisper()
        model_path = self._download_model()
        
        executable = self.whisper_path / 'main'
        if not executable.exists():
            raise RuntimeError("Failed to build whisper.cpp executable")
            
        self._whisper_installation = WhisperInstallation(
            base_path=self.whisper_path,
            executable_path=executable,
            model_path=model_path
        )
        
        return self._whisper_installation

    def process_audio(self, audio_file: Union[str, Path], output_file: Optional[Union[str, Path]] = None) -> str:
        """
        Process an audio file to generate viseme timings.
        
        Args:
            audio_file: Path to the input audio file
            output_file: Optional path for the output timing file. If None, creates one
                        based on the input filename.
                        
        Returns:
            Path to the output timing file
        """
        audio_file = Path(audio_file)
        if not audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
            
        if output_file is None:
            output_file = audio_file.with_suffix('.timing')
        output_file = Path(output_file)
        
        # Ensure whisper is installed
        install = self.ensure_installed()
        
        # Create a temporary file for whisper output
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.txt') as tmp_file:
            # Run whisper.cpp
            logger.info("Processing audio with whisper.cpp...")
            command = [
                str(install.executable_path),
                "-m", str(install.model_path),
                "-f", str(audio_file),
                "-ml", "1"  # Enable word-level timestamps
            ]
            
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Extract timestamp lines
            timestamp_lines = []
            timestamp_started = False
            for line in result.stdout.split('\n'):
                if line.startswith('['):
                    timestamp_started = True
                if timestamp_started and line.strip():
                    timestamp_lines.append(line)
            
            # Convert timestamps to visemes
            converter = TimestampToVisemeConverter()
            converter.process_input('\n'.join(timestamp_lines))
            
            # Write output
            with open(output_file, 'w') as f:
                f.write(converter.output_json())
            
            logger.info(f"Created viseme timings in {output_file}")
            return str(output_file)

# Copy the existing TimestampToVisemeConverter class here unmodified
@dataclass
class WordTiming:
    start_time: float
    end_time: float
    word: str

class AudioProcessor:
    def __init__(self, whisper_path: str, model_path: str):
        self.whisper_path = whisper_path
        self.model_path = model_path

    def process_audio(self, audio_file: str) -> str:
        """
        Process audio file using Whisper CPP and return the timestamp output
        """
        # Create command for Whisper CPP
        command = [
            self.whisper_path,
            "-m", self.model_path,
            "-f", audio_file,
            "-ml", "1"  # Enable word-level timestamps
        ]

        try:
            # Run Whisper CPP and capture output
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Extract the timestamp portion from the output
            # Skip the initial setup logs
            output_lines = result.stdout.split('\n')
            timestamp_lines = []
            timestamp_started = False
            
            for line in output_lines:
                if line.startswith('['):
                    timestamp_started = True
                if timestamp_started and line.strip():
                    timestamp_lines.append(line)
            
            return '\n'.join(timestamp_lines)
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Error running Whisper CPP: {e.stderr}")
        except Exception as e:
            raise RuntimeError(f"Error processing audio: {str(e)}")

class TimestampToVisemeConverter:
    # Phoneme to viseme mapping based on the provided table
    PHONEME_TO_VISEME = {
        # Consonants
        'b': 'p',   # bed
        'd': 't',   # dig
        'dZ': 'S',  # jump (d͡ʒ)
        'D': 'T',   # then (ð)
        'f': 'f',   # five
        'g': 'k',   # game
        'h': 'k',   # house
        'j': 'i',   # yes
        'k': 'k',   # cat
        'l': 't',   # lay
        'l=': 't',  # battle (syllabic l)
        'm': 'p',   # mouse
        'm=': 'p',  # anthem (syllabic m)
        'n': 't',   # nap
        'n=': 't',  # button (syllabic n)
        'N': 'k',   # thing (ŋ)
        'p': 'p',   # pin
        'r\\': 'r', # red (ɹ)
        's': 's',   # seem
        'S': 'S',   # ship (ʃ)
        't': 't',   # task
        'tS': 'S',  # chart (t͡ʃ)
        'T': 'T',   # thin (Θ)
        'v': 'f',   # vest
        'w': 'u',   # west
        'z': 's',   # zero
        'Z': 'S',   # vision (ʒ)
        
        # Vowels
        '@': '@',   # arena (ə)
        '@U': '@',  # goat (əʊ)
        '{': 'a',   # trap (æ)
        'aI': 'a',  # price (aɪ)
        'aU': 'a',  # mouth (aʊ)
        'A:': 'a',  # father (ɑː)
        'eI': 'e',  # face (eɪ)
        '3:': 'E',  # nurse (ɜː)
        'E': 'E',   # dress (ɛ)
        'E@': 'E',  # square (ɛə)
        'i': 'i',   # fleece (i:)
        'I': 'i',   # kit (ɪ)
        'I@': 'i',  # near (ɪə)
        'O:': 'O',  # thought (ɔː)
        'OI': 'O',  # choice (ɔɪ)
        'Q': 'O',   # lot (ɒ)
        'u:': 'u',  # goose
        'U': 'u',   # foot (ʊ)
        'U@': 'u',  # cure (ʊə)
        'V': 'E',   # strut (ʌ)
        
        # Additional mappings for common ASCII representations
        'CH': 'S',  # chart
        'SH': 'S',  # ship
        'TH': 'T',  # thin
        'DH': 'T',  # then
        'NG': 'k',  # thing
        'Y': 'i',   # yes
        
        # Common vowel representations
        'AA': 'a',  # father
        'AE': 'a',  # trap
        'AH': 'E',  # strut
        'AO': 'O',  # thought
        'AW': 'a',  # mouth
        'AY': 'a',  # price
        'EH': 'E',  # dress
        'ER': 'E',  # nurse
        'EY': 'e',  # face
        'IH': 'i',  # kit
        'IY': 'i',  # fleece
        'OW': 'O',  # goat
        'OY': 'O',  # choice
        'UH': 'u',  # foot
        'UW': 'u',  # goose
    }

    def __init__(self):
        self.word_timings: List[WordTiming] = []
        self.viseme_timings: List[Dict] = []
        
    # Add a new method for better word-to-phoneme conversion
    def word_to_phonemes(self, word: str) -> List[str]:
        """
        Convert a word to its phoneme sequence using improved rules.
        """
        word = word.lower().strip()
        phonemes = []
        i = 0
        
        # Common word-initial patterns
        if word.startswith('kn'):  # knight, know
            phonemes.append('n')
            i += 2
        elif word.startswith('wr'):  # write, wrong
            phonemes.append('r\\')
            i += 2
        elif word.startswith('ps'):  # psychology
            phonemes.append('s')
            i += 2
            
        while i < len(word):
            # Handle multi-character patterns
            if i < len(word) - 1:
                digraph = word[i:i+2]
                
                # Common vowel digraphs
                if digraph in {
                    'ee': ['i'],      # feet
                    'ea': ['i'],      # beat
                    'ai': ['eI'],     # wait
                    'ay': ['eI'],     # way
                    'oa': ['@U'],     # boat
                    'ow': ['@U'],     # low
                    'oo': ['u:'],     # boot
                    'ou': ['aU'],     # out
                    'au': ['O:'],     # august
                    'aw': ['O:'],     # law
                    'oi': ['OI'],     # coin
                    'oy': ['OI'],     # boy
                }.items():
                    if word[i:i+2] == digraph:
                        phonemes.extend(phones)
                        i += 2
                        continue
                
                # Common consonant digraphs
                elif digraph in {
                    'th': ['T'],      # thin
                    'ch': ['tS'],     # chain
                    'sh': ['S'],      # ship
                    'ph': ['f'],      # phone
                    'wh': ['w'],      # when
                    'ng': ['N'],      # sing
                    'ck': ['k'],      # back
                }.items():
                    if word[i:i+2] == digraph:
                        phonemes.extend(phones)
                        i += 2
                        continue
            
            # Handle single characters
            if word[i] in 'aeiou':
                if word[i] == 'a':
                    phonemes.append('{')
                elif word[i] == 'e':
                    phonemes.append('E')
                elif word[i] == 'i':
                    phonemes.append('I')
                elif word[i] == 'o':
                    phonemes.append('Q')
                elif word[i] == 'u':
                    phonemes.append('V')
            else:
                # Consonant mappings
                consonant_map = {
                    'b': ['b'],
                    'c': ['k'],  # Simplified - should check following vowel
                    'd': ['d'],
                    'f': ['f'],
                    'g': ['g'],
                    'h': ['h'],
                    'j': ['dZ'],
                    'k': ['k'],
                    'l': ['l'],
                    'm': ['m'],
                    'n': ['n'],
                    'p': ['p'],
                    'q': ['k'],  # Simplified
                    'r': ['r\\'],
                    's': ['s'],
                    't': ['t'],
                    'v': ['v'],
                    'w': ['w'],
                    'x': ['k', 's'],
                    'y': ['j'],
                    'z': ['z'],
                }
                if word[i] in consonant_map:
                    phonemes.extend(consonant_map[word[i]])
            i += 1
            
        return phonemes

    def word_to_visemes(self, word: WordTiming) -> List[Dict]:
        """Convert a word timing to a list of viseme timings with improved timing distribution."""
        visemes = []
        
        if not word.word or word.word.strip() in ['', ',', '.', ' ']:
            # Add silence marker
            visemes.append({
                "time": int(word.start_time * 1000),
                "type": "viseme",
                "value": "sil"
            })
            return visemes

        # Add word marker
        visemes.append({
            "time": int(word.start_time * 1000),
            "type": "word",
            "value": word.word,
            "start": int(word.start_time * 1000),
            "end": int(word.end_time * 1000)
        })

        # Get phonemes using improved method
        phonemes = self.word_to_phonemes(word.word)
        
        if not phonemes:
            return visemes

        # Calculate timing with improved distribution
        duration = word.end_time - word.start_time
        
        # Minimum viseme duration (50ms)
        min_duration = 0.05
        
        # Distribute time among phonemes
        if len(phonemes) > 0:
            # Allow for some overlap between visemes
            overlap_factor = 0.8
            base_duration = (duration * overlap_factor) / len(phonemes)
            
            # Ensure minimum duration
            time_per_phoneme = max(base_duration, min_duration)
            
            # Add visemes for each phoneme
            current_time = word.start_time
            prev_viseme = None
            
            for phoneme in phonemes:
                viseme = self.PHONEME_TO_VISEME.get(phoneme, self.PHONEME_TO_VISEME.get(phoneme.upper(), 'sil'))
                
                # Don't add consecutive duplicate visemes
                if viseme != prev_viseme:
                    visemes.append({
                        "time": int(current_time * 1000),
                        "type": "viseme",
                        "value": viseme
                    })
                    prev_viseme = viseme
                
                current_time += time_per_phoneme

        # Add final silence if there's a gap before the next word
        if visemes[-1]["type"] != "viseme" or visemes[-1]["value"] != "sil":
            visemes.append({
                "time": int(word.end_time * 1000),
                "type": "viseme",
                "value": "sil"
            })

        return visemes

    def parse_timestamp_lines(self, lines: List[str]) -> List[WordTiming]:
        """
        Parse multiple timestamp lines and combine split words.
        """
        pattern = r'\[(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})\]\s*(\S*)'
        timings = []
        current_word_parts = []
        current_start_time = None
        current_end_time = None

        def timestamp_to_seconds(ts: str) -> float:
            h, m, s = ts.split(':')
            return float(h) * 3600 + float(m) * 60 + float(s)

        def should_combine(prev_part: str, next_part: str) -> bool:
            # Rules for when parts should be combined
            if not prev_part or not next_part:
                return False
            
            # If one part is a single letter and the next part starts with a vowel
            if len(prev_part) == 1 and next_part[0].lower() in 'aeiou':
                return True
                
            # If parts are very close in time (less than 100ms apart)
            return True

        for line in lines:
            match = re.match(pattern, line.strip())
            if not match:
                continue

            start_str, end_str, word_part = match.groups()
            start_time = timestamp_to_seconds(start_str)
            end_time = timestamp_to_seconds(end_str)
            word_part = word_part.strip()

            if not word_part:
                continue

            # If this is the start of a new word
            if not current_word_parts or not should_combine(current_word_parts[-1], word_part):
                if current_word_parts:
                    # Save the previous word
                    timings.append(WordTiming(
                        start_time=current_start_time,
                        end_time=current_end_time,
                        word=''.join(current_word_parts).lower()
                    ))
                # Start a new word
                current_word_parts = [word_part]
                current_start_time = start_time
                current_end_time = end_time
            else:
                # Combine with current word
                current_word_parts.append(word_part)
                current_end_time = end_time

        # Don't forget the last word
        if current_word_parts:
            timings.append(WordTiming(
                start_time=current_start_time,
                end_time=current_end_time,
                word=''.join(current_word_parts).lower()
            ))

        return timings

    def process_input(self, input_text: str):
        """Process the input text with improved word combination."""
        lines = input_text.strip().split('\n')
        
        # Parse all lines with the new method
        word_timings = self.parse_timestamp_lines(lines)
        
        all_visemes = []
        
        # Add initial silence
        all_visemes.append({
            "time": 0,
            "type": "viseme",
            "value": "sil"
        })
        
        # Process each word timing
        for timing in word_timings:
            visemes = self.word_to_visemes(timing)
            all_visemes.extend(visemes)
        
        # Sort by time and remove duplicates
        all_visemes.sort(key=lambda x: x["time"])
        
        # Remove consecutive duplicate visemes
        filtered_visemes = []
        prev_viseme = None
        for v in all_visemes:
            if v["type"] == "word" or (
                v["type"] == "viseme" and 
                (not prev_viseme or 
                prev_viseme["type"] != "viseme" or 
                prev_viseme["value"] != v["value"])
            ):
                filtered_visemes.append(v)
                prev_viseme = v
        
        self.viseme_timings = filtered_visemes

    def output_json(self) -> str:
        """Output the viseme timings as JSON strings."""
        return '\n'.join(json.dumps(viseme) for viseme in self.viseme_timings)

def process_audio_file(audio_file: Union[str, Path], 
                      output_file: Optional[Union[str, Path]] = None,
                      installation_path: Optional[Union[str, Path]] = None) -> str:
    """
    Convenience function to process a single audio file.
    
    Args:
        audio_file: Path to the input audio file
        output_file: Optional path for the output timing file
        installation_path: Optional path for whisper.cpp installation
        
    Returns:
        Path to the output timing file
    """
    processor = VisemeProcessor(installation_path)
    return processor.process_audio(audio_file, output_file)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process audio file to generate viseme timings')
    parser.add_argument('audio_file', help='Input audio file')
    parser.add_argument('--output', help='Output timing file (optional)')
    parser.add_argument('--install-path', help='Installation path for whisper.cpp (optional)')
    
    args = parser.parse_args()
    
    try:
        output_path = process_audio_file(
            args.audio_file,
            args.output,
            args.install_path
        )
        print(f"Successfully created viseme timings: {output_path}")
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)
