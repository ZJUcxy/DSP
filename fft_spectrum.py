"""
从指定文件夹读取数据，截取行范围，带通滤波后输出 FFT 频谱（横坐标为频率）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, welch

SUPPORTED_SUFFIXES = {".csv", ".txt", ".dat", ".tsv", ".xlsx", ".xls"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="读取指定文件夹数据，截取行范围，滤波后绘制 FFT 频谱。"
    )
    parser.add_argument(
        "--folder",
        type=Path,
        required=True,
        help="数据所在文件夹",
    )
    parser.add_argument(
        "--file",
        type=str,
        default="",
        help="文件夹内的文件名。不指定则处理该文件夹下全部支持的数据文件",
    )
    parser.add_argument(
        "--start-row",
        type=int,
        default=0,
        help="起始行（从 0 开始，不含表头）",
    )
    parser.add_argument(
        "--end-row",
        type=int,
        default=-1,
        help="终止行（不含该行；-1 表示到文件末尾）",
    )
    parser.add_argument(
        "--fs",
        type=float,
        required=True,
        help="采样频率 Hz",
    )
    parser.add_argument(
        "--f-low",
        type=float,
        default=0.0,
        help="滤波器下限频率 Hz。0 表示不设下限（低通或全通）",
    )
    parser.add_argument(
        "--f-high",
        type=float,
        default=0.0,
        help="滤波器上限频率 Hz。0 表示不设上限（高通或全通）",
    )
    parser.add_argument(
        "--filter-order",
        type=int,
        default=4,
        help="Butterworth 滤波器阶数",
    )
    parser.add_argument(
        "--column",
        type=int,
        default=-1,
        help="数据列索引。-1 表示自动选择第一列数值列",
    )
    parser.add_argument(
        "--header",
        type=int,
        default=0,
        help="表头行号。无表头时设为 -1",
    )
    parser.add_argument(
        "--method",
        choices=["fft", "welch"],
        default="fft",
        help="频谱方法：fft 为单边幅值谱，welch 为功率谱密度",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("output"),
        help="频谱图和 CSV 输出目录",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="只保存结果，不弹窗显示",
    )
    return parser.parse_args()


def list_data_files(folder: Path, filename: str) -> list[Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"文件夹不存在: {folder}")

    if filename:
        path = folder / filename
        if not path.is_file():
            raise FileNotFoundError(f"文件不存在: {path}")
        return [path]

    files = sorted(
        p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not files:
        raise FileNotFoundError(
            f"{folder} 中没有支持的数据文件: {sorted(SUPPORTED_SUFFIXES)}"
        )
    return files


def load_table(path: Path, header: int) -> pd.DataFrame:
    suffix = path.suffix.lower()
    header_arg: int | None = None if header < 0 else header

    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, header=header_arg)

    sep = "\t" if suffix == ".tsv" else None
    return pd.read_csv(path, header=header_arg, sep=sep, engine="python")


def extract_signal(df: pd.DataFrame, column: int, start_row: int, end_row: int) -> np.ndarray:
    numeric = df.apply(pd.to_numeric, errors="coerce")

    if column >= 0:
        if column >= numeric.shape[1]:
            raise ValueError(f"列索引超出范围: {column}，共 {numeric.shape[1]} 列")
        series = numeric.iloc[:, column]
    else:
        valid_cols = [i for i in range(numeric.shape[1]) if numeric.iloc[:, i].notna().any()]
        if not valid_cols:
            raise ValueError("未找到可用的数值列")
        series = numeric.iloc[:, valid_cols[0]]

    n = len(series)
    if start_row < 0 or start_row >= n:
        raise ValueError(f"起始行无效: {start_row}，数据共 {n} 行")

    stop = n if end_row < 0 else end_row
    if stop <= start_row:
        raise ValueError(f"终止行必须大于起始行: start={start_row}, end={stop}")
    if stop > n:
        stop = n

    signal = series.iloc[start_row:stop].to_numpy(dtype=float)
    if np.isnan(signal).any():
        nan_count = int(np.isnan(signal).sum())
        raise ValueError(f"选取区间内存在 {nan_count} 个非数值，请调整行范围或数据列")
    return signal


def band_filter(
    signal: np.ndarray,
    fs: float,
    f_low: float,
    f_high: float,
    order: int,
) -> np.ndarray:
    nyquist = fs / 2.0
    if nyquist <= 0:
        raise ValueError("采样频率必须大于 0")

    low = f_low if f_low > 0 else None
    high = f_high if f_high > 0 else None

    if low is None and high is None:
        return signal.copy()

    if low is not None and low >= nyquist:
        raise ValueError(f"下限频率 {low} Hz 必须小于奈奎斯特频率 {nyquist} Hz")
    if high is not None and high >= nyquist:
        raise ValueError(f"上限频率 {high} Hz 必须小于奈奎斯特频率 {nyquist} Hz")
    if low is not None and high is not None and low >= high:
        raise ValueError(f"滤波器带宽无效: f_low={low}, f_high={high}")

    if low is not None and high is not None:
        sos = butter(order, [low / nyquist, high / nyquist], btype="bandpass", output="sos")
    elif high is not None:
        sos = butter(order, high / nyquist, btype="lowpass", output="sos")
    else:
        sos = butter(order, low / nyquist, btype="highpass", output="sos")

    return sosfiltfilt(sos, signal)


def compute_fft(signal: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    n = len(signal)
    windowed = signal * np.hanning(n)
    spectrum = np.fft.rfft(windowed)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    # 单边幅值谱，补偿汉宁窗相干增益
    amplitude = np.abs(spectrum) * 2.0 / np.sum(np.hanning(n))
    if n % 2 == 0:
        amplitude[-1] /= 2.0
    amplitude[0] /= 2.0
    return freqs, amplitude


def compute_welch(signal: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    nperseg = min(4096, len(signal))
    freqs, psd = welch(signal, fs=fs, window="hann", nperseg=nperseg)
    return freqs, psd


def plot_and_save(
    freqs: np.ndarray,
    values: np.ndarray,
    title: str,
    ylabel: str,
    out_png: Path,
    out_csv: Path,
    show: bool,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"frequency_hz": freqs, "value": values}).to_csv(out_csv, index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(freqs, values, color="#1f77b4", linewidth=1.0)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    if show:
        plt.show()
    else:
        plt.close()


def process_file(path: Path, args: argparse.Namespace) -> None:
    df = load_table(path, args.header)
    signal = extract_signal(df, args.column, args.start_row, args.end_row)
    filtered = band_filter(signal, args.fs, args.f_low, args.f_high, args.filter_order)

    if args.method == "fft":
        freqs, values = compute_fft(filtered, args.fs)
        ylabel = "Amplitude"
    else:
        freqs, values = compute_welch(filtered, args.fs)
        ylabel = "PSD"

    band_text = (
        f"band {args.f_low:g}-{args.f_high:g} Hz"
        if args.f_low > 0 or args.f_high > 0
        else "no filter"
    )
    title = (
        f"{path.name} | rows {args.start_row}:{args.end_row if args.end_row >= 0 else 'end'} | "
        f"fs={args.fs:g} Hz | {band_text}"
    )

    stem = path.stem
    out_png = args.out_dir / f"{stem}_spectrum.png"
    out_csv = args.out_dir / f"{stem}_spectrum.csv"
    plot_and_save(freqs, values, title, ylabel, out_png, out_csv, show=not args.no_show)
    print(f"完成: {path.name} -> {out_png} , {out_csv}")


def main() -> int:
    args = parse_args()
    try:
        files = list_data_files(args.folder, args.file)
        for path in files:
            process_file(path, args)
    except Exception as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
