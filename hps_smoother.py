import math
from collections import deque

class ParameterSmoother:
    """
    Smooths audio spatial parameters over time using exponential moving average
    or linear interpolation between frames.
    """
    
    def __init__(self, alpha=0.3, history_size=5, method='ema'):
        """
        Args:
            alpha: Smoothing factor for EMA (0-1). Lower = smoother but more latency.
                   0.3 means 30% new value, 70% old value.
            history_size: Number of past frames to keep for interpolation methods.
            method: 'ema' (exponential moving average) or 'lerp' (linear interpolation)
        """
        self.alpha = alpha
        self.history_size = history_size
        self.method = method
        self.history = {}  # key -> deque of past values
        
    def smooth_value(self, key, value):
        """Smooth a single scalar value."""
        if key not in self.history:
            self.history[key] = deque(maxlen=self.history_size)
            self.history[key].append(value)
            return value
        
        if self.method == 'ema':
            # Exponential moving average
            prev = self.history[key][-1]
            smoothed = self.alpha * value + (1 - self.alpha) * prev
        elif self.method == 'lerp':
            # Linear interpolation with last N values
            hist = list(self.history[key])
            weights = [i / len(hist) for i in range(1, len(hist) + 1)]
            weight_sum = sum(weights)
            smoothed = sum(h * w for h, w in zip(hist, weights)) / weight_sum
            smoothed = self.alpha * value + (1 - self.alpha) * smoothed
        else:
            smoothed = value
        
        self.history[key].append(smoothed)
        return smoothed
    
    def smooth_angle(self, key, angle, wrap_range=None):
        """
        Smooth angular values (IPD) with proper wrapping.
        
        Args:
            key: Identifier for this angle stream
            angle: Current angle value
            wrap_range: Tuple (min, max) for wrapping, e.g., (-π, π)
        """
        if wrap_range:
            min_val, max_val = wrap_range
            angle_range = max_val - min_val
        else:
            angle_range = 2 * math.pi
            min_val = -math.pi
        
        if key not in self.history:
            self.history[key] = deque(maxlen=self.history_size)
            self.history[key].append(angle)
            return angle
        
        prev = self.history[key][-1]
        
        # Handle angle wrapping - find shortest path
        diff = angle - prev
        if abs(diff) > angle_range / 2:
            if diff > 0:
                angle -= angle_range
            else:
                angle += angle_range
        
        if self.method == 'ema':
            smoothed = self.alpha * angle + (1 - self.alpha) * prev
        else:
            hist = list(self.history[key])
            weights = [i / len(hist) for i in range(1, len(hist) + 1)]
            weight_sum = sum(weights)
            smoothed = sum(h * w for h, w in zip(hist, weights)) / weight_sum
            smoothed = self.alpha * angle + (1 - self.alpha) * smoothed
        
        # Wrap back to range
        if wrap_range:
            while smoothed < min_val:
                smoothed += angle_range
            while smoothed >= max_val:
                smoothed -= angle_range
        
        self.history[key].append(smoothed)
        return smoothed
    
    def smooth_mps_band(self, band_idx, pan, ipd, ic):
        """
        Smooth MPS band parameters.
        
        Args:
            band_idx: Band index identifier
            pan, ipd, ic: Current frame parameters
            
        Returns:
            Tuple of (smoothed_pan, smoothed_ipd, smoothed_ic)
        """
        pan_key = f"mps_pan_{band_idx}"
        ipd_key = f"mps_ipd_{band_idx}"
        ic_key = f"mps_ic_{band_idx}"
        
        smoothed_pan = self.smooth_value(pan_key, pan)
        smoothed_ipd = self.smooth_angle(ipd_key, ipd, wrap_range=(-math.pi, math.pi))
        
        # IC is boolean - use majority voting or threshold
        if ic_key not in self.history:
            self.history[ic_key] = deque(maxlen=self.history_size)
        
        self.history[ic_key].append(1.0 if ic else 0.0)
        ic_avg = sum(self.history[ic_key]) / len(self.history[ic_key])
        smoothed_ic = ic_avg > 0.5
        
        return smoothed_pan, smoothed_ipd, smoothed_ic
    
    def smooth_mps_results(self, results):
        """
        Smooth entire MPS results list.
        
        Args:
            results: List of (band_idx, pan, ipd, ic) tuples
            
        Returns:
            Smoothed list in same format
        """
        smoothed = []
        for band_idx, pan, ipd, ic in results:
            s_pan, s_ipd, s_ic = self.smooth_mps_band(band_idx, pan, ipd, ic)
            smoothed.append((band_idx, s_pan, s_ipd, s_ic))
        return smoothed
    
    def smooth_phsc_harmonic(self, obj_idx, harm_idx, params):
        """
        Smooth PHSC harmonic parameters.
        
        Args:
            obj_idx: Object index
            harm_idx: Harmonic index
            params: Dict with keys 'IID', 'IPD', 'ICC', 'ICLD'
            
        Returns:
            Smoothed params dict
        """
        smoothed = {}
        
        for param_name, value in params.items():
            key = f"phsc_{obj_idx}_{harm_idx}_{param_name}"
            
            if param_name == 'IPD':
                # IPD is angular
                smoothed[param_name] = self.smooth_angle(key, value, wrap_range=(-10.0, 10.0))
            else:
                smoothed[param_name] = self.smooth_value(key, value)
        
        return smoothed
    
    def smooth_phsc_objects(self, harmonic_objects):
        """
        Smooth entire PHSC harmonic objects list.
        
        Args:
            harmonic_objects: List of dicts with 'freq' and 'harmonics'
            
        Returns:
            Smoothed objects in same format
        """
        smoothed_objects = []
        
        for obj_idx, obj in enumerate(harmonic_objects):
            smoothed_harmonics = []
            
            for harm_idx, h in enumerate(obj['harmonics']):
                smoothed_h = self.smooth_phsc_harmonic(obj_idx, harm_idx, h)
                smoothed_harmonics.append(smoothed_h)
            
            smoothed_objects.append({
                'freq': obj['freq'],
                'harmonics': smoothed_harmonics
            })
        
        return smoothed_objects
    
    def reset(self):
        """Clear all history."""
        self.history.clear()
    
    def reset_key(self, key_pattern):
        """Clear history for keys matching a pattern."""
        keys_to_remove = [k for k in self.history.keys() if key_pattern in k]
        for k in keys_to_remove:
            del self.history[k]

