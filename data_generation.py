"""
Synthetic data generation for Etherscan Pipeline
"""

import numpy as np
import setigen as stg
from astropy import units as u
from typing import Tuple, Dict, Optional
import logging
from random import random
import gc
from multiprocessing import Pool, cpu_count, shared_memory

logger = logging.getLogger(__name__)

# Global variables to store shared memory reference for multiprocessing workers
# This avoids duplicating large arrays across worker processes
_GLOBAL_SHM = None
_GLOBAL_BACKGROUNDS = None
_GLOBAL_SHAPE = None
_GLOBAL_DTYPE = None

def _init_worker(shm_name, shape, dtype):
    """
    Initialize worker process with shared memory reference
    This avoids copying data to each worker, saving memory

    Args:
        shm_name: Name of the shared memory block
        shape: Shape of the background array
        dtype: Data type of the background array

    Note:
        Worker cleanup is automatic - when the pool terminates, the OS reclaims
        all worker process resources including shared memory file descriptors.
        Only the main process needs to unlink() the shared memory block.
    """
    global _GLOBAL_SHM, _GLOBAL_BACKGROUNDS, _GLOBAL_SHAPE, _GLOBAL_DTYPE

    # Attach to existing shared memory block
    _GLOBAL_SHM = shared_memory.SharedMemory(name=shm_name)

    # Create numpy array view of shared memory (no copy!)
    _GLOBAL_BACKGROUNDS = np.ndarray(shape, dtype=dtype, buffer=_GLOBAL_SHM.buf)
    _GLOBAL_SHAPE = shape
    _GLOBAL_DTYPE = dtype


def log_norm(data: np.ndarray) -> np.ndarray:
    """
    Apply log normalization to data
    """
    # Add small epsilon to avoid log(0)
    data = data + 1e-10

    # Transform data into log-space
    data = np.log(data)
    # Shift data to be >= 0
    data = data - data.min()
    # Normalize data to [0, 1]
    if data.max() > 0:
        data = data / data.max()

    return data


# NOTE: not 100% sure how this function works. ported from Peter's code. comments added by Claude. assuming it works as intended?
# NOTE: verify that we're randomly drawing a combo of snr, drift_rate, and signal_width for each injection?
def new_cadence(data: np.ndarray, snr: float, width_bin: int,
                freq_resolution: float, time_resolution: float) -> Tuple[np.ndarray, float, float]:
    """
    Inject a single drifting narrowband signal into a stacked cadence array
    """
    # Set noise parameter (for simulating randomness in drift rate calculation)
    noise = 3

    # Randomly select a starting frequency bin (channel) to start the signal injection
    # Avoids edges (bin 0)
    starting_bin = int(random() * (width_bin - 1)) + 1

    # Get the total number of time samples in stacked array (typically 96 for 6 obs x 16 time bins)
    total_time = data.shape[0]

    # Randomly select a positive or negative drift direction
    if np.random.choice([-1, 1]) > 0:
        # Positive drift
        slope_pixel = (total_time / starting_bin)  # Signal drifts upward in frequency
        # Convert from pixel space to physical units by multiplying by time_resolution / freq_resolution ratio
        # Then add random noise to make drift rates more realistic
        slope_physical = (slope_pixel) * (time_resolution / freq_resolution) + random() * noise
    else:
        # Negative drift
        slope_pixel = (total_time / (starting_bin - width_bin))  # Signal drifts downward in frequency
        # Convert from pixel space to physical units by multiplying by time_resolution / freq_resolution ratio
        # Then add random noise to make drift rates more realistic
        slope_physical = (slope_pixel) * (time_resolution / freq_resolution) - random() * noise

    # Convert slope to drift rate
    drift_rate = -1 * (1 / slope_physical)

    # Calculate signal width (in Hz)
    # Base random component: 0-50 Hz
    # Add component proportional to drift rate magnitude to keep signal coherent
    signal_width = random() * 50 + abs(drift_rate) * 18. / 1

    # Calculate y-intercept for linear signal trajectory
    y_intercept = total_time - slope_pixel * (starting_bin)

    # Create setigen Frame
    frame = stg.Frame.from_data(
        df=freq_resolution*u.Hz,
        dt=time_resolution*u.s,
        fch1=0*u.MHz,  # Set reference frequency (center frequency offset)
        data=data,
        ascending=True  # Frequency increases with channel index
    )

    # Inject signal
    signal = frame.add_signal(
        # Use linear drift trajectory starting at starting_bin & with the calculated drift rate
        stg.constant_path(
            f_start=frame.get_frequency(index=starting_bin),
            drift_rate=drift_rate*u.Hz/u.s
        ),
        # Constant intensity over time, calibrated to achieve target snr
        stg.constant_t_profile(level=frame.get_intensity(snr=snr)),
        # Gaussian shape in frequency domain with calculated signal width
        stg.gaussian_f_profile(width=signal_width*u.Hz),
        # Constant bandpass profile (no frequency-dependent scaling)
        stg.constant_bp_profile(level=1)
    )

    # Extract the modified data (with signal injection) from the setigen Frame
    modified_data = frame.data.copy()

    # Return the modified data array, slope (in pixel coordinates), and y-intercept
    return modified_data, slope_pixel, y_intercept


def check_valid_intersection(slope_1, slope_2, intercept_1, intercept_2):
    """
    Check if 2 drifting signals intersect in the ON regions
    """
    x_intersect = (intercept_2 - intercept_1) / (slope_1 - slope_2)
    y_intersect = slope_1 * x_intersect + intercept_1

    on_y_coords = [(0, 16), (32, 48), (64, 80)]
    for y_lower, y_upper in on_y_coords:
        if y_lower <= y_intersect <= y_upper:
            return False
    return True


def create_false(plate: np.ndarray, snr_base: float, snr_range: float,
                 width_bin: int, freq_resolution: float, time_resolution: float,
                 inject: bool = True, dynamic_range: Optional[float] = None) -> np.ndarray:
    """
    Create false signal class
    If specified, RFI is injected into all 6 observations. Otherwise, no RFI is injected
    """
    # Select random background from plate
    background_index = int(plate.shape[0] * random())
    base = plate[background_index, :, :, :]

    # Initialize empty output array
    n_obs = plate.shape[1]
    n_time = plate.shape[2]
    final = np.zeros((n_obs, n_time, width_bin))

    # Inject RFI into all 6 observations
    if inject:
        # Prepare data for signal injection by stacking all 6 observations vertically
        # (6, 16, 512) -> (96, 512)
        # Obs 0: rows 0-15, Obs 1: rows 16-31, Obs 2: rows 32-47, ...
        data = np.zeros((n_obs * n_time, width_bin))
        for i in range(n_obs):
            data[i*n_time:(i+1)*n_time, :] = base[i, :, :]

        # Select a random SNR from the given range & inject RFI into all 6 observations
        snr = random() * snr_range + snr_base
        cadence, _, _ = new_cadence(data, snr, width_bin, freq_resolution, time_resolution)

        # Reshape stacked data back into original shape & log-normalize after signal injection
        for i in range(n_obs):
            final[i, :, :] = log_norm(cadence[i*n_time:(i+1)*n_time, :])

    # Just return background. No signal injection
    else:
        # Log-normalize base background
        for i in range(n_obs):
            final[i, :, :] = log_norm(base[i, :, :])

    return final


def create_true_single(plate: np.ndarray, snr_base: float, snr_range: float,
                       width_bin: int, freq_resolution: float, time_resolution: float,
                       inject: Optional[bool] = None, dynamic_range: Optional[float] = None) -> np.ndarray:
    """
    Create true-single signal class
    ETI signal is injected into the ON observations only
    """
    # Select random background from plate
    background_index = int(plate.shape[0] * random())
    base = plate[background_index, :, :, :]

    # Initialize empty output array
    n_obs = plate.shape[1]
    n_time = plate.shape[2]
    final = np.zeros((n_obs, n_time, width_bin))

    # Prepare data for signal injection by stacking all 6 observations vertically
    # (6, 16, 512) -> (96, 512)
    # Obs 0: rows 0-15, Obs 1: rows 16-31, Obs 2: rows 32-47, ...
    data = np.zeros((n_obs * n_time, width_bin))
    for i in range(n_obs):
        data[i*n_time:(i+1)*n_time, :] = base[i, :, :]

    # Select a random SNR from the given range & inject RFI
    snr = random() * snr_range + snr_base
    cadence, _, _ = new_cadence(data, snr, width_bin, freq_resolution, time_resolution)

    # Reshape stacked data back into original shape & log-normalize after signal injection
    for i in range(n_obs):
        if i % 2 == 0:
            # ONs: injected signal
            final[i, :, :] = log_norm(cadence[i*n_time:(i+1)*n_time, :])
        else:
            # OFFs: original background
            final[i, :, :] = log_norm(data[i*n_time:(i+1)*n_time, :])

    return final


def create_true_double(plate: np.ndarray, snr_base: float, snr_range: float,
                       width_bin: int, freq_resolution: float, time_resolution: float,
                       inject: Optional[bool] = None, dynamic_range: float = 1) -> np.ndarray:
    """
    Create true-double signal class 
    Non-intersecting ETI & RFI signals are injected into ON-only & ON-OFF, respectively
    """
    # Select random background from plate
    background_index = int(plate.shape[0] * random())
    base = plate[background_index, :, :, :]

    # Initialize empty output array
    n_obs = plate.shape[1]
    n_time = plate.shape[2]
    final = np.zeros((n_obs, n_time, width_bin))

    # Prepare data for signal injection by stacking all 6 observations vertically
    # (6, 16, 512) -> (96, 512)
    # Obs 0: rows 0-15, Obs 1: rows 16-31, Obs 2: rows 32-47, ...
    data = np.zeros((n_obs * n_time, width_bin))
    for i in range(n_obs):
        data[i*n_time:(i+1)*n_time, :] = base[i, :, :]

    # Select a random SNR from the given range
    snr = random() * snr_range + snr_base

    # Retry signal injection until we get valid non-intersecting signals
    while True:
        # Inject RFI
        cadence_1, slope_1, intercept_1 = new_cadence(data, snr, width_bin, freq_resolution, time_resolution)
        # Inject ETI
        cadence_2, slope_2, intercept_2 = new_cadence(cadence_1, snr*dynamic_range, width_bin, freq_resolution, time_resolution)

        if slope_1 != slope_2 and check_valid_intersection(slope_1, slope_2, intercept_1, intercept_2):
            break

    # Reshape stacked data back into original shape & log-normalize after signal injection
    for i in range(n_obs):
        if i % 2 == 0:
            # ONs: 2 injected signals (ETI + RFI)
            final[i, :, :] = log_norm(cadence_2[i*n_time:(i+1)*n_time, :])
        else:
            # OFFs: 1 injected signal (RFI only)
            final[i, :, :] = log_norm(cadence_1[i*n_time:(i+1)*n_time, :])

    return final


def _single_cadence_wrapper(args):
    """
    Wrapper function for multiprocessing that unpacks arguments and generates a single cadence
    Uses global background plates to avoid serialization overhead

    Args:
        args: Tuple of (function, snr_base, snr_range, width_bin, freq_resolution, time_resolution, inject, dynamic_range)

    Returns:
        Single cadence array of shape (6, 16, width_bin)
    """
    global _GLOBAL_BACKGROUNDS
    function, snr_base, snr_range, width_bin, freq_resolution, time_resolution, inject, dynamic_range = args
    return function(_GLOBAL_BACKGROUNDS, snr_base=snr_base, snr_range=snr_range, width_bin=width_bin,
                   freq_resolution=freq_resolution, time_resolution=time_resolution,
                   inject=inject, dynamic_range=dynamic_range)


def batch_create_cadence(function, samples: int, plate: np.ndarray,
                        snr_base: int = 10, snr_range: float = 40, width_bin: int = 512,
                        freq_resolution: float = 2.7939677238464355, time_resolution: float = 18.25361108,
                        inject: Optional[bool] = None, dynamic_range: Optional[float] = None,
                        pool: Optional[Pool] = None) -> np.ndarray:
    """
    Batch wrapper for creating multiple cadences using multiprocessing

    Args:
        function: Cadence generation function (create_false, create_true_single, create_true_double)
        samples: Number of cadences to generate
        plate: Background plate array (only used if pool is None)
        snr_base: Base SNR value
        snr_range: SNR range for randomization
        width_bin: Number of frequency bins
        freq_resolution: Frequency resolution in Hz
        time_resolution: Time resolution in seconds
        inject: Whether to inject signals (for create_false)
        dynamic_range: Dynamic range for signal injection (for create_true_double)
        pool: Pre-initialized multiprocessing Pool (if None, runs sequentially)

    Returns:
        Array of shape (samples, 6, 16, width_bin) containing generated cadences
    """
    # Pre-allocate output array
    cadence = np.zeros((samples, 6, 16, width_bin))

    if pool is None:
        # Sequential execution (backwards compatibility)
        for i in range(samples):
            cadence[i, :, :, :] = function(plate, snr_base=snr_base, snr_range=snr_range, width_bin=width_bin,
                                          freq_resolution=freq_resolution, time_resolution=time_resolution,
                                          inject=inject, dynamic_range=dynamic_range)
    else:
        # Parallel execution using provided pool
        # Prepare arguments for each parallel task (no plate - uses global)
        args_list = [
            (function, snr_base, snr_range, width_bin, freq_resolution, time_resolution, inject, dynamic_range)
            for _ in range(samples)
        ]

        # Calculate optimal chunksize for load balancing
        # Aim for ~4 chunks per worker to balance overhead vs parallelism
        try:
            n_workers = pool._processes
        except AttributeError:
            n_workers = cpu_count()
        chunksize = max(1, samples // (n_workers * 4))

        # Use pool to generate cadences in parallel
        results = pool.map(_single_cadence_wrapper, args_list, chunksize=chunksize)

        # Collect results into pre-allocated array
        for i, result in enumerate(results):
            cadence[i, :, :, :] = result

    return cadence


class DataGenerator:
    """Synthetic data generator"""

    def __init__(self, config, background_plates: np.ndarray, n_processes: Optional[int] = None):
        """
        Initialize generator

        Args:
            config: Configuration object
            background_plates: Array of background observations
                              Shape: (n_backgrounds, 6, 16, 512) after preprocessing
            n_processes: Number of parallel processes for signal injection (defaults to cpu_count())
        """
        self.config = config

        # Sanity check: verify no NaN or Inf values in background plates
        if np.isnan(background_plates).any():
            raise ValueError("background_plates contains NaN values")
        if np.isinf(background_plates).any():
            raise ValueError("background_plates contains Inf values")

        self.n_backgrounds = len(background_plates)
        self._background_shape = background_plates.shape 
        self._background_dtype = background_plates.dtype

        # Sanity check: verify downsampling working as expected
        width_bin_downsampled = config.data.width_bin // config.data.downsample_factor

        if self._background_shape[3] != width_bin_downsampled:
            raise ValueError(f"Expected {width_bin_downsampled} channels. Got {self._background_shape[3]} instead")

        self.width_bin = width_bin_downsampled
        self.freq_resolution = self.config.data.freq_resolution
        self.time_resolution = self.config.data.time_resolution

        # Set number of processes for multiprocessing
        self.n_processes = n_processes if n_processes is not None else cpu_count()

        # Create persistent process pool for efficient parallel execution
        # Use shared memory to avoid duplicating background data across workers
        if self.n_processes > 1:
            # Create shared memory block for background data
            nbytes = background_plates.nbytes
            self.shm = shared_memory.SharedMemory(create=True, size=nbytes)

            # Copy background data into shared memory
            shared_array = np.ndarray(
                self._background_shape,
                dtype=self._background_dtype,
                buffer=self.shm.buf
            )
            shared_array[:] = background_plates[:]
            self.backgrounds = shared_array

            # Create pool with shared memory reference instead of data copy
            self.pool = Pool(
                processes=self.n_processes,
                initializer=_init_worker,
                initargs=(self.shm.name, self._background_shape, self._background_dtype)
            )

            logger.info(f"Created multiprocessing pool with {self.n_processes} workers using shared memory")
            logger.info(f"Shared memory size: {nbytes / 1e9:.2f} GB (shared across all workers)")
        else:
            self.shm = None
            self.pool = None
            self.backgrounds = background_plates
            logger.info("Running in sequential mode (n_processes=1)")

        logger.info(f"DataGenerator initialized with {self.n_backgrounds} background plates")
        logger.info(f"Background shape: {self._background_shape}")

    def close(self):
        """Explicitly close the multiprocessing pool and shared memory"""
        # Close pool first
        if hasattr(self, 'pool') and self.pool is not None:
            try:
                self.pool.close()
                self.pool.join()
            except Exception:
                # Ignore errors during cleanup (e.g., if called during interpreter shutdown)
                pass
            finally:
                self.pool = None

        # Clean up shared memory
        if hasattr(self, 'shm') and self.shm is not None:
            try:
                self.shm.close()
                self.shm.unlink()  # Delete shared memory block
                logger.info("Shared memory cleaned up successfully")
            except Exception as e:
                # Log but don't raise - might already be cleaned up
                logger.warning(f"Error cleaning up shared memory: {e}")
            finally:
                self.shm = None

    def __del__(self):
        """Clean up multiprocessing pool and shared memory on deletion"""
        # Try to close pool and shared memory, but don't raise errors during garbage collection
        try:
            self.close()
        except Exception:
            # Ignore all errors during __del__ to avoid issues during interpreter shutdown
            pass

    def generate_train_batch(self, n_samples: int, snr_base: int, snr_range: int) -> Dict[str, np.ndarray]:
        """
        Generate training batch using chunking & multiprocessing

        main: collapsed cadences 
          - total: n_samples
          - split: 1/4 balanced between false-no-signal, false-with-rfi, true-single, true-double
        false: non-collapsed false cadences 
          - total: n_samples 
          - split: 1/2 balanced between false-no-signal, false-with-rfi
        true: non-collapsed true cadences 
          - total: n_samples 
          - split: 1/2 balanced between true-single, true-double
        """
        max_chunk_size = self.config.training.signal_injection_chunk_size
        n_chunks = max(1, (n_samples + max_chunk_size - 1) // max_chunk_size)
        
        logger.info(f"Generating {n_samples} samples in {n_chunks} chunks of max {max_chunk_size}")

        # Pre-allocate output arrays
        all_main = np.empty((n_samples, 6, 16, self.width_bin), dtype=np.float32)
        all_false = np.empty((n_samples, 6, 16, self.width_bin), dtype=np.float32)
        all_true = np.empty((n_samples, 6, 16, self.width_bin), dtype=np.float32)
        
        for chunk_idx in range(n_chunks):
            chunk_size = min(max_chunk_size, n_samples - chunk_idx * max_chunk_size)
            if chunk_size <= 0:
                break

            start_idx = chunk_idx * max_chunk_size
            end_idx = start_idx + chunk_size
                
            logger.info(f"Generating chunk {chunk_idx + 1}/{n_chunks} with {chunk_size} samples")
            
            # Split chunk into equal partitions (for balanced classes)
            quarter = max(1, chunk_size // 4)
            half = max(1, chunk_size // 2)
            
            # Pure background
            quarter_false_no_signal = batch_create_cadence(
                create_false, quarter, self.backgrounds,
                snr_base=snr_base,
                snr_range=snr_range,
                width_bin=self.width_bin,
                freq_resolution=self.freq_resolution,
                time_resolution=self.time_resolution,
                inject=False,
                pool=self.pool
            )

            # RFI only
            quarter_false_with_rfi = batch_create_cadence(
                create_false, quarter, self.backgrounds,
                snr_base=snr_base,
                snr_range=snr_range,
                width_bin=self.width_bin,
                freq_resolution=self.freq_resolution,
                time_resolution=self.time_resolution,
                inject=True,
                pool=self.pool
            )

            # ETI only
            quarter_true_single = batch_create_cadence(
                create_true_single, quarter, self.backgrounds,
                snr_base=snr_base,
                snr_range=snr_range,
                width_bin=self.width_bin,
                freq_resolution=self.freq_resolution,
                time_resolution=self.time_resolution,
                pool=self.pool
            )

            # ETI + RFI
            quarter_true_double = batch_create_cadence(
                create_true_double, quarter, self.backgrounds,
                snr_base=snr_base,
                snr_range=snr_range,
                width_bin=self.width_bin,
                freq_resolution=self.freq_resolution,
                time_resolution=self.time_resolution,
                dynamic_range=1,
                pool=self.pool
            )
            
            # Concatenate for main training data (collapsed cadences)
            chunk_main = np.concatenate([
                quarter_false_no_signal, quarter_false_with_rfi, quarter_true_single, quarter_true_double
            ], axis=0)
            
            # Generate separate true/false non-collapsed cadences for training set diversity
            # Used to calculate clustering loss & train RF
            half_false_no_signal = batch_create_cadence(
                create_false, half, self.backgrounds,
                snr_base=snr_base,
                snr_range=snr_range,
                width_bin=self.width_bin,
                freq_resolution=self.freq_resolution,
                time_resolution=self.time_resolution,
                inject=False,
                pool=self.pool
            )

            half_false_with_rfi = batch_create_cadence(
                create_false, half, self.backgrounds,
                snr_base=snr_base,
                snr_range=snr_range,
                width_bin=self.width_bin,
                freq_resolution=self.freq_resolution,
                time_resolution=self.time_resolution,
                inject=True,
                pool=self.pool
            )

            half_true_single = batch_create_cadence(
                create_true_single, half, self.backgrounds,
                snr_base=snr_base,
                snr_range=snr_range,
                width_bin=self.width_bin,
                freq_resolution=self.freq_resolution,
                time_resolution=self.time_resolution,
                pool=self.pool
            )

            half_true_double = batch_create_cadence(
                create_true_double, half, self.backgrounds,
                snr_base=snr_base,
                snr_range=snr_range,
                width_bin=self.width_bin,
                freq_resolution=self.freq_resolution,
                time_resolution=self.time_resolution,
                dynamic_range=1,
                pool=self.pool
            )

            chunk_false = np.concatenate([
                half_false_no_signal, half_false_with_rfi
            ], axis=0)

            chunk_true = np.concatenate([
                half_true_single, half_true_double
            ], axis=0)

            # Store chunks directly into output array
            all_main[start_idx:end_idx] = chunk_main
            all_false[start_idx:end_idx] = chunk_false
            all_true[start_idx:end_idx] = chunk_true
            
            # Clean up chunk data immediately
            del quarter_false_no_signal, quarter_false_with_rfi, quarter_true_single, quarter_true_double
            del half_false_no_signal, half_false_with_rfi, half_true_single, half_true_double
            del chunk_main, chunk_false, chunk_true
            gc.collect()
            
            logger.info(f"Chunk {chunk_idx + 1} complete, memory cleared")
        
        # Create result dictionary with references to pre-allocated arrays
        result = {
            'concatenated': all_main,
            'false': all_false,
            'true': all_true
        }

        # Sanity check: verify post-injection data normalization
        for key in ['concatenated', 'false', 'true']:
            min_val = np.min(result[key])
            max_val = np.max(result[key])
            mean_val = np.mean(result[key])
            logger.info(f"Post-injection {key} stats: min={min_val:.6f}, max={max_val:.6f}, mean={mean_val:.6f}")
            if max_val > 1.0:
                logger.error(f"Post-injection {key} values too large! Max: {max_val}")
                raise ValueError(f"Post-injection {key} normalization check failed")
            elif min_val < 0.0: 
                logger.error(f"Post-injection {key} values too small! Min: {min_val}")
                raise ValueError(f"Post-injection {key} normalization check failed")
            elif np.isnan(result[key]).any():
                logger.error(f"Post-injection {key} contains NaN values!")
                raise ValueError(f"Post-injection {key} normalization check failed")
            elif np.isinf(result[key]).any():
                logger.error(f"Post-injection {key} contains Inf values!")
                raise ValueError(f"Post-injection {key} normalization check failed")
            else:
                logger.info(f"Post-injection {key} data properly normalized")
        
        return result
