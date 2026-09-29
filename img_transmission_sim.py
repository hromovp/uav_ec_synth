import numpy as np
from PIL import Image
from scipy.signal import butter, filtfilt

# ============================================================
# NTSC CONSTANTS
# ============================================================
SAMPLING_RATE = 20e6  # Sampling rate
SUBCARRIER_FREQ = 3.579545e6  # Subcarrier frequency
RF_CARRIER = 5.5e9  # True RF carrier, used only analytically (never sampled directly)

LINE_TIME = 63.556e-6
SYNC_TIME = 4.7e-6
BACK_PORCH = 5.8e-6
ACTIVE_TIME = 52.66e-6
LUMA_GAIN = 0.4

SYNC_LEVEL = -0.4
CHROMA_AMPL = 0.3
BURST_AMPL = 0.3
PEDESTAL = 1.0

# ============================================================
# TWO RAY GROUND MODEL CONSTANTS
# ============================================================
SPEED_OF_LIGHT = 3e8
TX_ANTENNA_GAIN_DBI = 10.0
RX_ANTENNA_GAIN_DBI = 21.0
TX_HEIGHT_M = 10.0
RX_HEIGHT_M = 2.0
LINK_DISTANCE_M = 3000.0
GROUND_REFLECTION_COEFF = -1.0  # ideal ground (perfect reflector)
NOMINAL_SNR_DB = 50  # SNR at gain factor of 1, i.e. with no path loss

# Sample counts
SYNC_SAMPLES = int(SAMPLING_RATE * SYNC_TIME)
BP_SAMPLES = int(SAMPLING_RATE * BACK_PORCH)
ACTIVE_SAMPLES = int(SAMPLING_RATE * ACTIVE_TIME)
LINE_SAMPLES = int(SAMPLING_RATE * LINE_TIME)
BURST_SAMPLES = int(SAMPLING_RATE * 9 / SUBCARRIER_FREQ)


# ============================================================
# COLOR SPACE
# ============================================================
def rgb_to_yiq(rgb):
    M = np.array([
        [0.299, 0.587, 0.114],
        [0.596, -0.275, -0.321],
        [0.212, -0.523, 0.311]
    ])
    return rgb @ M.T


def yiq_to_rgb(Y, I, Q):
    R = Y + 0.956 * I + 0.621 * Q
    G = Y - 0.272 * I - 0.647 * Q
    B = Y - 1.106 * I + 1.703 * Q
    return np.clip(np.stack([R, G, B], axis=-1), 0, 1)


# ============================================================
# FILTERS
# ============================================================
def lowpass(x, cutoff):
    b, a = butter(5, cutoff / (SAMPLING_RATE / 2))
    return filtfilt(b, a, x)


def bandpass(x, f1, f2):
    b, a = butter(5, [f1 / (SAMPLING_RATE / 2), f2 / (SAMPLING_RATE / 2)], btype='band')
    return filtfilt(b, a, x)


# ============================================================
# NTSC BASEBAND ENCODER
# ============================================================
def encode_ntsc_line(Y, I, Q):
    t = np.arange(LINE_SAMPLES) / SAMPLING_RATE
    sig = np.zeros(LINE_SAMPLES)

    # Sync
    sig[:SYNC_SAMPLES] = SYNC_LEVEL

    # Color burst
    tb = t[SYNC_SAMPLES:SYNC_SAMPLES + BURST_SAMPLES]
    sig[SYNC_SAMPLES:SYNC_SAMPLES + BURST_SAMPLES] += (
            BURST_AMPL * np.sin(2 * np.pi * SUBCARRIER_FREQ * tb)
    )

    # Active video
    start = SYNC_SAMPLES + BP_SAMPLES
    end = start + ACTIVE_SAMPLES
    t_act = t[start:end]

    x_old = np.linspace(0, 1, len(Y))
    x_new = np.linspace(0, 1, ACTIVE_SAMPLES)

    Yl = np.interp(x_new, x_old, Y)
    Il = np.interp(x_new, x_old, I)
    Ql = np.interp(x_new, x_old, Q)

    chroma = Il * np.cos(2 * np.pi * SUBCARRIER_FREQ * t_act) \
             + Ql * np.sin(2 * np.pi * SUBCARRIER_FREQ * t_act)

    sig[start:end] += Yl + CHROMA_AMPL * chroma
    return sig


# ============================================================
# TRANSMISSION CHANNEL (antenna gain + two-ray fading + noise)
# ============================================================
def two_ray_gain(distance, tx_height, rx_height, carrier_freq, gamma=-1.0):
    # Flat-fading magnitude of the two-ray ground-reflection channel (valid under envelope detection).
    wavelength = SPEED_OF_LIGHT / carrier_freq
    d1 = np.sqrt(distance ** 2 + (tx_height - rx_height) ** 2)
    d2 = np.sqrt(distance ** 2 + (tx_height + rx_height) ** 2)
    # complex/phasor form (dropping the constant E_0d_0, 
    # since we only care about the relative attenuation-and-interference 
    # gain applied to our baseband signal, not the absolute field strength)
    h = (1 / d1) * np.exp(-1j * 2 * np.pi * d1 / wavelength) \
        + gamma * (1 / d2) * np.exp(-1j * 2 * np.pi * d2 / wavelength)
    return np.abs(h)


def rf_channel(sig, noise_power):
    # fixed absolute noise floor, independent of the (possibly attenuated) signal power
    return sig + np.sqrt(noise_power) * np.random.randn(len(sig))


# ============================================================
# NTSC DECODER
# ============================================================
def decode_ntsc_line(rx):
    start = SYNC_SAMPLES + BP_SAMPLES
    end = start + ACTIVE_SAMPLES
    active = rx[start:end]

    # Black restore
    black = np.mean(rx[SYNC_SAMPLES + 10:SYNC_SAMPLES + 50])
    active = (active - black) / 0.5

    # Luma
    Y = lowpass(active, 3.0e6)
    Y *= LUMA_GAIN
    # Chroma
    chroma = bandpass(active, 2.5e6, 4.2e6)
    t = np.arange(len(chroma)) / SAMPLING_RATE

    I = lowpass(chroma * np.cos(2 * np.pi * SUBCARRIER_FREQ * t), 1.3e6)
    Q = lowpass(chroma * np.sin(2 * np.pi * SUBCARRIER_FREQ * t), 0.6e6)

    # Burst phase correction
    burst = rx[SYNC_SAMPLES:SYNC_SAMPLES + BURST_SAMPLES]
    tb = np.arange(len(burst)) / SAMPLING_RATE
    phase = np.angle(np.mean(burst * np.exp(-1j * 2 * np.pi * SUBCARRIER_FREQ * tb)))

    Icorr = I * np.cos(phase) - Q * np.sin(phase)
    Qcorr = I * np.sin(phase) + Q * np.cos(phase)

    return Y, Icorr, Qcorr


# ============================================================
# IMAGE UTILITIES
# ============================================================
def resample(line, width):
    return np.interp(
        np.linspace(0, len(line) - 1, width),
        np.arange(len(line)),
        line
    )


# ============================================================
# JAMMING
# ============================================================
def barrage_jammer(rf, jnr_db=10):
    """
    Wideband noise barrage jammer.
    jnr_db = jammer-to-signal power ratio (dB)
    jnr_db = 0 → annoying
    jnr_db = 10 → severe
    jnr_db = 20 → unusable
    """
    sig_power = np.mean(rf ** 2)
    jam_power = sig_power * (10 ** (jnr_db / 10))
    noise = np.sqrt(jam_power) * np.random.randn(len(rf))
    return rf + noise


def single_tone_jammer(rf, fs, f_jam, amplitude=0.3, phase=0.0):
    """
    Continuous-wave single-tone jammer.
    f_jam near RF_CARRIER causes severe interference.

    f_jam = RF_CARRIER → carrier override
    f_jam = RF_CARRIER ± 1–2 MHz → diagonal bars
    f_jam = RF_CARRIER ± SUBCARRIER_FREQ → chroma chaos
    """
    t = np.arange(len(rf)) / fs
    jammer = amplitude * np.cos(2 * np.pi * f_jam * t + phase)
    return rf + jammer


def successive_pulse_jammer(
        rf,
        pulse_rate=500,  # pulses per second
        pulse_width=2e-6,  # seconds
        amplitude=5.0
):
    """
    Repeated RF pulse jammer.
    Higher pulse_rate → sync shredding
    Wider pulse_width → white bars
    Higher amplitude → instant loss of picture
    """
    fs = SAMPLING_RATE
    samples = len(rf)
    pulse_samples = int(pulse_width * fs)
    interval = int(fs / pulse_rate)

    jam = np.zeros_like(rf)

    for start in range(0, samples, interval):
        end = min(start + pulse_samples, samples)
        jam[start:end] = amplitude * np.random.randn(end - start)

    return rf + jam


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    WIDTH, HEIGHT = 640, 480

    img = Image.open("image1.jpg").resize((WIDTH, HEIGHT))
    rgb = np.asarray(img) / 255.0
    yiq = rgb_to_yiq(rgb.reshape(-1, 3)).reshape(rgb.shape)

    Yimg, Iimg, Qimg = [], [], []

    tx_gain = np.sqrt(10 ** (TX_ANTENNA_GAIN_DBI / 10))
    rx_gain = np.sqrt(10 ** (RX_ANTENNA_GAIN_DBI / 10))
    channel_gain = two_ray_gain(
        LINK_DISTANCE_M, TX_HEIGHT_M, RX_HEIGHT_M, RF_CARRIER, GROUND_REFLECTION_COEFF
    )
    total_gain = tx_gain * channel_gain * rx_gain

    for y in range(HEIGHT):
        broadband = encode_ntsc_line(
            yiq[y, :, 0],
            yiq[y, :, 1],
            yiq[y, :, 2]
        )

        # Normalize the broadband signal to prevent clipping and maintain consistent amplitude
        broadband /= max(np.max(np.abs(broadband)), 1e-6)

        # From SNR formula: noise_power = signal_power / (10^(SNR_dB/10))
        noise_power = np.mean(broadband ** 2) / (10 ** (NOMINAL_SNR_DB / 10))

        sig = broadband * total_gain
        sig = rf_channel(sig, noise_power)
        sig = sig / total_gain  # AGC: restore nominal amplitude, keeps the SNR degradation

        # Apply jamming
        # sig = barrage_jammer(sig, jnr_db=10)
        # sig = single_tone_jammer(sig, SAMPLING_RATE, RF_CARRIER + 1.5e6, amplitude=1.2)
        sig = successive_pulse_jammer(sig, pulse_rate=120000, amplitude=10, pulse_width=5e-6)

        Y, I, Q = decode_ntsc_line(sig)

        Yimg.append(resample(Y, WIDTH))
        Iimg.append(resample(I, WIDTH))
        Qimg.append(resample(Q, WIDTH))

    rgb_out = yiq_to_rgb(
        np.array(Yimg),
        np.array(Iimg),
        np.array(Qimg)
    )

    img_received = Image.fromarray((rgb_out * 255).astype(np.uint8))
    img_received.show()
    # img_received.save("reconstr1.png")
