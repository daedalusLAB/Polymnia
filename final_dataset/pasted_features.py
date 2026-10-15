import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.spatial import ConvexHull

def _process_single_sequence(
    sequence: pd.DataFrame,
    window_size: int,
    polyorder: int,
    threshold: float,
    fps: float
) -> pd.DataFrame:
    sequence = sequence.sort_values('frame').copy()

    if len(sequence) < 2:
        return sequence

    HAND_MAPPING = {
        'right': {'x': 'nx_4', 'y': 'ny_4'},
        'left':  {'x': 'nx_7', 'y': 'ny_7'},
    }

    # 1) SMOOTHING
    for hand in ['right', 'left']:
        xcol = HAND_MAPPING[hand]['x']
        ycol = HAND_MAPPING[hand]['y']

        wlen = min(window_size, len(sequence))
        if wlen % 2 == 0:
            wlen -= 1
        wlen = max(wlen, 3)

        safe_polyorder = min(polyorder, wlen - 1)

        smooth_x = savgol_filter(sequence[xcol], window_length=wlen, polyorder=safe_polyorder)
        smooth_y = savgol_filter(sequence[ycol], window_length=wlen, polyorder=safe_polyorder)

        sequence[f'{hand}_hand_x'] = smooth_x
        sequence[f'{hand}_hand_y'] = smooth_y

    # 2) FRAME DELTA (seconds per frame)
    dt = 1.0 / fps

    # 3) HAND-TO-HAND FEATURES
    sequence['hands_distance_euclidean'] = np.sqrt(
        (sequence['right_hand_x'] - sequence['left_hand_x'])**2 +
        (sequence['right_hand_y'] - sequence['left_hand_y'])**2
    )
    sequence['hands_distance_manhattan'] = (
        np.abs(sequence['right_hand_x'] - sequence['left_hand_x']) +
        np.abs(sequence['right_hand_y'] - sequence['left_hand_y'])
    )
    sequence['hands_angle_rad'] = np.arctan2(
        sequence['left_hand_y'] - sequence['right_hand_y'],
        sequence['left_hand_x'] - sequence['right_hand_x']
    )
    sequence['hands_angle_deg'] = np.degrees(sequence['hands_angle_rad'])

    # 4) PER-HAND FEATURES
    for hand in ['right', 'left']:
        xcol_new = f'{hand}_hand_x'
        ycol_new = f'{hand}_hand_y'

        dist_nose = np.sqrt(sequence[xcol_new]**2 + sequence[ycol_new]**2)
        sequence[f'{hand}_hand_nose_distance'] = dist_nose

        angle_rad = np.arctan2(sequence[ycol_new], sequence[xcol_new])
        sequence[f'{hand}_hand_nose_angle_rad'] = angle_rad
        sequence[f'{hand}_hand_nose_angle_deg'] = np.degrees(angle_rad)

        sequence[f'{hand}_hand_above_nose'] = sequence[ycol_new] < 0

        sequence[f'{hand}_hand_side'] = np.sign(sequence[xcol_new])

        sequence[f'{hand}_hand_quadrant'] = np.select(
            [
                (sequence[xcol_new] >= 0) & (sequence[ycol_new] >= 0),
                (sequence[xcol_new] <  0) & (sequence[ycol_new] >= 0),
                (sequence[xcol_new] <  0) & (sequence[ycol_new] <  0),
                (sequence[xcol_new] >= 0) & (sequence[ycol_new] <  0)
            ],
            [1, 2, 3, 4],
            default=np.nan
        )

        vx = np.gradient(sequence[xcol_new]) / dt
        vy = np.gradient(sequence[ycol_new]) / dt
        sequence[f'{hand}_hand_velocity_x'] = vx
        sequence[f'{hand}_hand_velocity_y'] = vy
        sequence[f'{hand}_hand_velocity_total'] = np.sqrt(vx**2 + vy**2)

        ax = np.gradient(vx) / dt
        ay = np.gradient(vy) / dt
        sequence[f'{hand}_hand_accel_x'] = ax
        sequence[f'{hand}_hand_accel_y'] = ay
        sequence[f'{hand}_hand_accel_total'] = np.sqrt(ax**2 + ay**2)

        sequence[f'{hand}_hand_zone'] = np.select(
            [
                (sequence[ycol_new] < -threshold) & (sequence[xcol_new] < -threshold),
                (sequence[ycol_new] < -threshold) & (abs(sequence[xcol_new]) <= threshold),
                (sequence[ycol_new] < -threshold) & (sequence[xcol_new] > threshold),

                (abs(sequence[ycol_new]) <= threshold) & (sequence[xcol_new] < -threshold),
                (abs(sequence[ycol_new]) <= threshold) & (abs(sequence[xcol_new]) <= threshold),
                (abs(sequence[ycol_new]) <= threshold) & (sequence[xcol_new] > threshold),

                (sequence[ycol_new] > threshold) & (sequence[xcol_new] < -threshold),
                (sequence[ycol_new] > threshold) & (abs(sequence[xcol_new]) <= threshold),
                (sequence[ycol_new] > threshold) & (sequence[xcol_new] > threshold),
            ],
            [7, 8, 9, 4, 5, 6, 1, 2, 3],
            default=5
        )

    # 4.5) FRAME-BY-FRAME DISTANCES
    dx_right = np.diff(sequence['right_hand_x'])
    dy_right = np.diff(sequence['right_hand_y'])
    dx_left  = np.diff(sequence['left_hand_x'])
    dy_left  = np.diff(sequence['left_hand_y'])

    right_distances = np.sqrt(dx_right**2 + dy_right**2)
    left_distances  = np.sqrt(dx_left**2 + dy_left**2)

    sequence['right_hand_frame_distance'] = np.concatenate(([0], right_distances))
    sequence['left_hand_frame_distance']  = np.concatenate(([0], left_distances))

    # 5) SYMMETRY, CENTROID, AREAS
    sequence['hands_vertical_symmetry']   = np.abs(sequence['right_hand_y'] + sequence['left_hand_y'])
    sequence['hands_horizontal_symmetry'] = np.abs(sequence['right_hand_x'] + sequence['left_hand_x'])
    sequence['hands_centroid_x'] = (sequence['right_hand_x'] + sequence['left_hand_x']) / 2.0
    sequence['hands_centroid_y'] = (sequence['right_hand_y'] + sequence['left_hand_y']) / 2.0

    sequence['gesture_area_circle'] = np.pi * (sequence['hands_distance_euclidean'] / 2.0)**2

    sequence['gesture_area_box'] = (
        np.abs(sequence['right_hand_x'] - sequence['left_hand_x']) *
        np.abs(sequence['right_hand_y'] - sequence['left_hand_y'])
    )

    # 6) ENERGY & COORDINATION
    sequence['right_hand_kinetic_energy'] = 0.5 * (sequence['right_hand_velocity_total']**2)
    sequence['left_hand_kinetic_energy']  = 0.5 * (sequence['left_hand_velocity_total']**2)
    sequence['hands_total_energy']        = sequence['right_hand_kinetic_energy'] + sequence['left_hand_kinetic_energy']

    eps = 1e-6
    rv = sequence['right_hand_velocity_total']
    lv = sequence['left_hand_velocity_total']
    sequence['hands_coordination_coef'] = (rv * lv) / (np.sqrt(rv**2 + lv**2) + eps)

    return sequence


def _aggregate_sequence_features(sequence: pd.DataFrame) -> pd.Series:
    def iqr(x: pd.Series) -> float:
        return x.quantile(0.75) - x.quantile(0.25)

    def sign_changes(arr: pd.Series) -> int:
        arr = arr.dropna()
        if arr.empty:
            return 0
        signs = np.sign(arr)
        return np.sum(np.diff(signs) != 0)

    def time_to_peak_velocity(vel_series: pd.Series) -> float:
        if vel_series.empty:
            return np.nan
        peak_idx = vel_series.idxmax()
        frames = sequence['frame']
        if peak_idx in frames.index:
            peak_frame = frames.loc[peak_idx]
            start_frame = frames.min()
            return peak_frame - start_frame
        else:
            return np.nan

    aggregator = {}

    aggregator['id']         = sequence['id'].iloc[0]
    aggregator['numeric_id'] = sequence['numeric_id'].iloc[0]
    aggregator['label']      = sequence['label'].iloc[0]

    for hand in ['right', 'left']:
        col = f'{hand}_hand_nose_distance'
        aggregator[f'{hand}_hand_nose_dist_min']    = sequence[col].min()
        aggregator[f'{hand}_hand_nose_dist_max']    = sequence[col].max()
        aggregator[f'{hand}_hand_nose_dist_mean']   = sequence[col].mean()
        aggregator[f'{hand}_hand_nose_dist_median'] = sequence[col].median()
        aggregator[f'{hand}_hand_nose_dist_std']    = sequence[col].std()

    aggregator['right_hand_x_range'] = sequence['right_hand_x'].max() - sequence['right_hand_x'].min()
    aggregator['right_hand_y_range'] = sequence['right_hand_y'].max() - sequence['right_hand_y'].min()
    aggregator['left_hand_x_range']  = sequence['left_hand_x'].max()  - sequence['left_hand_x'].min()
    aggregator['left_hand_y_range']  = sequence['left_hand_y'].max()  - sequence['left_hand_y'].min()

    v_thresh = 2.0

    for hand in ['right', 'left']:
        vt_col = f'{hand}_hand_velocity_total'
        aggregator[f'{hand}_hand_velocity_min']    = sequence[vt_col].min()
        aggregator[f'{hand}_hand_velocity_max']    = sequence[vt_col].max()
        aggregator[f'{hand}_hand_velocity_mean']   = sequence[vt_col].mean()
        aggregator[f'{hand}_hand_velocity_median'] = sequence[vt_col].median()
        aggregator[f'{hand}_hand_velocity_std']    = sequence[vt_col].std()
        aggregator[f'{hand}_hand_velocity_iqr']    = iqr(sequence[vt_col])
        aggregator[f'{hand}_frac_frames_fast']     = (sequence[vt_col] > v_thresh).mean()
        aggregator[f'{hand}_hand_time_to_peak_vel'] = time_to_peak_velocity(sequence[vt_col])

    aggregator['right_hand_total_distance'] = sequence['right_hand_frame_distance'].sum()
    aggregator['left_hand_total_distance']  = sequence['left_hand_frame_distance'].sum()
    aggregator['both_hands_total_distance'] = (
        aggregator['right_hand_total_distance'] +
        aggregator['left_hand_total_distance']
    )

    for hand in ['right', 'left']:
        at_col = f'{hand}_hand_accel_total'
        aggregator[f'{hand}_hand_accel_min']          = sequence[at_col].min()
        aggregator[f'{hand}_hand_accel_max']          = sequence[at_col].max()
        aggregator[f'{hand}_hand_accel_mean']         = sequence[at_col].mean()
        aggregator[f'{hand}_hand_accel_median']       = sequence[at_col].median()
        aggregator[f'{hand}_hand_accel_std']          = sequence[at_col].std()
        aggregator[f'{hand}_hand_accel_iqr']          = iqr(sequence[at_col])
        aggregator[f'{hand}_hand_accel_sign_changes'] = sign_changes(sequence[at_col])

    agg_ang = sequence['hands_angle_deg']
    aggregator['hands_angle_deg_min']    = agg_ang.min()
    aggregator['hands_angle_deg_max']    = agg_ang.max()
    aggregator['hands_angle_deg_mean']   = agg_ang.mean()
    aggregator['hands_angle_deg_median'] = agg_ang.median()
    aggregator['hands_angle_deg_std']    = agg_ang.std()
    aggregator['hands_angle_deg_range']  = agg_ang.max() - agg_ang.min()

    aggregator['right_hand_total_energy_sum'] = sequence['right_hand_kinetic_energy'].sum()
    aggregator['left_hand_total_energy_sum']  = sequence['left_hand_kinetic_energy'].sum()
    aggregator['hands_total_energy_sum']      = sequence['hands_total_energy'].sum()

    aggregator['right_hand_total_energy_max'] = sequence['right_hand_kinetic_energy'].max()
    aggregator['left_hand_total_energy_max']  = sequence['left_hand_kinetic_energy'].max()
    aggregator['hands_total_energy_max']      = sequence['hands_total_energy'].max()

    c_thresh = 0.8
    agg_coord = sequence['hands_coordination_coef']
    aggregator['hands_coordination_coef_mean'] = agg_coord.mean()
    aggregator['hands_coordination_coef_max']  = agg_coord.max()
    aggregator['hands_frac_frames_high_coord'] = (agg_coord > c_thresh).mean()

    hde = sequence['hands_distance_euclidean']
    aggregator['hands_distance_euclidean_mean']   = hde.mean()
    aggregator['hands_distance_euclidean_median'] = hde.median()
    aggregator['hands_distance_euclidean_max']    = hde.max()
    aggregator['hands_distance_euclidean_std']    = hde.std()

    gab = sequence['gesture_area_box']
    gac = sequence['gesture_area_circle']
    aggregator['gesture_area_box_max']    = gab.max()
    aggregator['gesture_area_box_mean']   = gab.mean()
    aggregator['gesture_area_circle_max'] = gac.max()
    aggregator['gesture_area_circle_mean']= gac.mean()

    if 'right_hand_distance_cog' in sequence.columns:
        rdc = sequence['right_hand_distance_cog']
        aggregator['right_hand_distance_cog_max']    = rdc.max()
        aggregator['right_hand_distance_cog_mean']   = rdc.mean()
        aggregator['right_hand_distance_cog_median'] = rdc.median()
        aggregator['right_hand_distance_cog_std']    = rdc.std()

    if 'left_hand_distance_cog' in sequence.columns:
        ldc = sequence['left_hand_distance_cog']
        aggregator['left_hand_distance_cog_max']     = ldc.max()
        aggregator['left_hand_distance_cog_mean']    = ldc.mean()
        aggregator['left_hand_distance_cog_median']  = ldc.median()
        aggregator['left_hand_distance_cog_std']     = ldc.std()

    aggregator['right_hand_total_distance'] = sequence['right_hand_frame_distance'].sum()
    aggregator['left_hand_total_distance']  = sequence['left_hand_frame_distance'].sum()
    aggregator['both_hands_total_distance'] = aggregator['right_hand_total_distance'] + aggregator['left_hand_total_distance']

    aggregator['hands_total_energy_sum']     = sequence['hands_total_energy'].sum()
    aggregator['right_hand_velocity_mean']   = sequence['right_hand_velocity_total'].mean()
    aggregator['left_hand_velocity_mean']    = sequence['left_hand_velocity_total'].mean()

    return pd.Series(aggregator)


def _compute_cog_each_hand(df: pd.DataFrame) -> pd.DataFrame:
    df_cog = (
        df.groupby('id', as_index=False)
          .agg({
             'right_hand_x': 'mean',
             'right_hand_y': 'mean',
             'left_hand_x':  'mean',
             'left_hand_y':  'mean'
          })
          .rename(columns={
             'right_hand_x': 'right_cog_x',
             'right_hand_y': 'right_cog_y',
             'left_hand_x':  'left_cog_x',
             'left_hand_y':  'left_cog_y'
          })
    )
    df_merged = df.merge(df_cog, on='id', how='left')
    return df_merged


def _compute_distance_to_cog(df: pd.DataFrame) -> pd.DataFrame:
    df['right_hand_distance_cog'] = np.sqrt(
        (df['right_hand_x'] - df['right_cog_x'])**2 +
        (df['right_hand_y'] - df['right_cog_y'])**2
    )
    df['left_hand_distance_cog'] = np.sqrt(
        (df['left_hand_x'] - df['left_cog_x'])**2 +
        (df['left_hand_y'] - df['left_cog_y'])**2
    )
    return df


def _compute_gesture_centroid(sequence: pd.DataFrame) -> dict:
    centroid_coords = {
        'right_hand_centroid_x': sequence['right_hand_x'].mean(),
        'right_hand_centroid_y': sequence['right_hand_y'].mean(),
        'left_hand_centroid_x': sequence['left_hand_x'].mean(),
        'left_hand_centroid_y': sequence['left_hand_y'].mean(),
    }
    return centroid_coords

def _compute_polar_coords_from_centroid(sequence: pd.DataFrame, centroids: dict) -> pd.DataFrame:
    for hand in ['right', 'left']:
        hand_x = sequence[f'{hand}_hand_x']
        hand_y = sequence[f'{hand}_hand_y']
        centroid_x = centroids[f'{hand}_hand_centroid_x']
        centroid_y = centroids[f'{hand}_hand_centroid_y']

        radius = np.sqrt((hand_x - centroid_x)**2 + (hand_y - centroid_y)**2)
        sequence[f'{hand}_hand_radius_from_centroid'] = radius

        angle = np.arctan2(hand_y - centroid_y, hand_x - centroid_x)
        sequence[f'{hand}_hand_angle_from_centroid'] = angle

    return sequence


def _bin_relative_distances(
    df: pd.DataFrame,
    method: str = 'quartiles',
    bins_right=None,
    bins_left=None
) -> pd.DataFrame:
    if method == 'quartiles':
        df['right_hand_distance_cog_qbin'] = pd.qcut(
            df['right_hand_distance_cog'],
            q=4,
            labels=['1', '2', '3', '4']
        )
    elif method == 'fixed' and bins_right is not None:
        labels = [f'bin{i}' for i in range(1, len(bins_right))]
        df['right_hand_distance_cog_bin'] = pd.cut(
            df['right_hand_distance_cog'],
            bins=bins_right,
            labels=labels,
            include_lowest=True
        )

    if method == 'quartiles':
        df['left_hand_distance_cog_qbin'] = pd.qcut(
            df['left_hand_distance_cog'],
            q=4,
            labels=['1', '2', '3', '4']
        )
    elif method == 'fixed' and bins_left is not None:
        labels = [f'bin{i}' for i in range(1, len(bins_left))]
        df['left_hand_distance_cog_bin'] = pd.cut(
            df['left_hand_distance_cog'],
            bins=bins_left,
            labels=labels,
            include_lowest=True
        )
    return df


def _build_extra_features(sequence: pd.DataFrame) -> pd.Series:
    extra = {}

    for hand in ['right', 'left']:
        x = sequence[f'{hand}_hand_x']
        y = sequence[f'{hand}_hand_y']

        radius = np.sqrt(x**2 + y**2)
        theta  = np.arctan2(y, x)

        extra[f'{hand}_radius_max']    = radius.max()
        extra[f'{hand}_radius_min']    = radius.min()
        extra[f'{hand}_radius_mean']   = radius.mean()
        extra[f'{hand}_radius_median'] = radius.median()
        extra[f'{hand}_radius_std']    = radius.std()
        extra[f'{hand}_radius_iqr']    = radius.quantile(0.75) - radius.quantile(0.25)

        cos_mean = np.mean(np.cos(theta))
        sin_mean = np.mean(np.sin(theta))
        circ_mean = np.arctan2(sin_mean, cos_mean)
        extra[f'{hand}_theta_circ_mean'] = np.degrees(circ_mean)

        R_bar = np.sqrt(cos_mean**2 + sin_mean**2)
        circ_std = np.sqrt(-2 * np.log(R_bar + 1e-8))
        extra[f'{hand}_theta_circ_std'] = np.degrees(circ_std)

        for R in [3.5, 4.0]:
            mask_ext = radius > R
            extra[f'{hand}_count_radius_gt_{R}'] = mask_ext.sum()
            extra[f'{hand}_frac_radius_gt_{R}']  = mask_ext.mean()

        dx = x.max() - x.min()
        dy = y.max() - y.min()
        extra[f'{hand}_trajectory_span'] = np.sqrt(dx**2 + dy**2)

        pts = np.column_stack((x, y))
        try:
            if len(np.unique(pts, axis=0)) >= 3:
                hull = ConvexHull(pts)
                extra[f'{hand}_hull_area'] = hull.volume
            else:
                extra[f'{hand}_hull_area'] = 0.0
        except Exception:
            extra[f'{hand}_hull_area'] = np.nan

        total_dist = sequence[f'{hand}_hand_frame_distance'].sum()
        if len(x) >= 2:
            straight = np.sqrt((x.iloc[-1] - x.iloc[0])**2 + (y.iloc[-1] - y.iloc[0])**2)
        else:
            straight = np.nan
        extra[f'{hand}_path_ratio'] = total_dist / (straight + 1e-8)

    sep = sequence['hands_distance_euclidean']
    extra['hands_sep_max']    = sep.max()
    extra['hands_sep_min']    = sep.min()
    extra['hands_sep_mean']   = sep.mean()
    extra['hands_sep_median'] = sep.median()
    extra['hands_sep_std']    = sep.std()
    extra['hands_sep_iqr']    = sep.quantile(0.75) - sep.quantile(0.25)

    for D in [3.5, 4.0]:
        mask_wide = sep > D
        extra[f'count_hands_sep_gt_{D}'] = mask_wide.sum()
        extra[f'frac_hands_sep_gt_{D}']  = mask_wide.mean()

    if 'right_hand_distance_cog' in sequence.columns:
        for hand in ['right', 'left']:
            cog_col = f'{hand}_hand_distance_cog'
            dist_cog = sequence[cog_col]
            extra[f'{hand}_cog_dist_max']    = dist_cog.max()
            extra[f'{hand}_cog_dist_mean']   = dist_cog.mean()
            extra[f'{hand}_cog_dist_iqr']    = dist_cog.quantile(0.75) - dist_cog.quantile(0.25)
            for C in [3.0]:
                mask_far = dist_cog > C
                extra[f'{hand}_count_far_cog_gt_{C}'] = mask_far.sum()
                extra[f'{hand}_frac_far_cog_gt_{C}']  = mask_far.mean()

    for hand in ['right', 'left']:
        vel = sequence[f'{hand}_hand_velocity_total']
        acc = sequence[f'{hand}_hand_accel_total']

        try:
            seq_vel_q = pd.qcut(vel, q=4, labels=[1, 2, 3, 4])
            for q in [1, 2, 3, 4]:
                extra[f'{hand}_vel_q{q}_count'] = (seq_vel_q == q).sum()
                extra[f'{hand}_vel_q{q}_frac']  = (seq_vel_q == q).mean()
        except Exception:
            for q in [1, 2, 3, 4]:
                extra[f'{hand}_vel_q{q}_count'] = 0
                extra[f'{hand}_vel_q{q}_frac']  = 0.0

        try:
            seq_acc_q = pd.qcut(acc, q=4, labels=[1, 2, 3, 4])
            for q in [1, 2, 3, 4]:
                extra[f'{hand}_acc_q{q}_count'] = (seq_acc_q == q).sum()
                extra[f'{hand}_acc_q{q}_frac']  = (seq_acc_q == q).mean()
        except Exception:
            for q in [1, 2, 3, 4]:
                extra[f'{hand}_acc_q{q}_count'] = 0
                extra[f'{hand}_acc_q{q}_frac']  = 0.0

    for hand in ['right', 'left']:
        quad_col = f'{hand}_hand_quadrant'
        for q in [1, 2, 3, 4]:
            extra[f'{hand}_quad_{q}_count'] = (sequence[quad_col] == q).sum()
            extra[f'{hand}_quad_{q}_frac']  = (sequence[quad_col] == q).mean()

    for hand in ['right', 'left']:
        zone_col = f'{hand}_hand_zone'
        for z in range(1, 10):
            extra[f'{hand}_zone_{z}_count'] = (sequence[zone_col] == z).sum()
            extra[f'{hand}_zone_{z}_frac']  = (sequence[zone_col] == z).mean()

    cx = sequence['hands_centroid_x']
    cy = sequence['hands_centroid_y']
    centroid_radius = np.sqrt(cx**2 + cy**2)
    extra['hands_centroid_radius_max']    = centroid_radius.max()
    extra['hands_centroid_radius_mean']   = centroid_radius.mean()
    extra['hands_centroid_radius_iqr']    = centroid_radius.quantile(0.75) - centroid_radius.quantile(0.25)

    for Cc in [3.0]:
        mask_cent_far = centroid_radius > Cc
        extra[f'count_centroid_radius_gt_{Cc}'] = mask_cent_far.sum()
        extra[f'frac_centroid_radius_gt_{Cc}']  = mask_cent_far.mean()

    for hand in ['right', 'left']:
        vx = sequence[f'{hand}_hand_velocity_x']
        vy = sequence[f'{hand}_hand_velocity_y']
        extra[f'{hand}_vel_sign_changes_x'] = np.sum(np.diff(np.sign(vx.fillna(0))) != 0)
        extra[f'{hand}_vel_sign_changes_y'] = np.sum(np.diff(np.sign(vy.fillna(0))) != 0)

        ax = sequence[f'{hand}_hand_accel_x']
        ay = sequence[f'{hand}_hand_accel_y']
        extra[f'{hand}_acc_sign_changes_x'] = np.sum(np.diff(np.sign(ax.fillna(0))) != 0)
        extra[f'{hand}_acc_sign_changes_y'] = np.sum(np.diff(np.sign(ay.fillna(0))) != 0)

    for hand in ['right', 'left']:
        radius = np.sqrt(sequence[f'{hand}_hand_x']**2 + sequence[f'{hand}_hand_y']**2)
        for R in [3.5]:
            mask_R = radius > R
            if mask_R.any():
                idx = mask_R.idxmax()
                extra[f'{hand}_first_frame_radius_gt_{R}'] = sequence.loc[idx, 'frame'] - sequence['frame'].min()
            else:
                extra[f'{hand}_first_frame_radius_gt_{R}'] = np.nan

    for hand in ['right', 'left']:
        x = sequence[f'{hand}_hand_x'].values
        y = sequence[f'{hand}_hand_y'].values
        points = np.column_stack((x, y))
        if len(points) >= 2:
            cov = np.cov(points, rowvar=False)
            eigvals, _ = np.linalg.eig(cov)
            extra[f'{hand}_ellipse_area'] = np.pi * np.sqrt(eigvals[0] + 1e-8) * np.sqrt(eigvals[1] + 1e-8)
        else:
            extra[f'{hand}_ellipse_area'] = np.nan

    centroids = _compute_gesture_centroid(sequence)
    sequence_with_polar = _compute_polar_coords_from_centroid(sequence.copy(), centroids)

    for hand in ['right', 'left']:
        radius_data = sequence_with_polar[f'{hand}_hand_radius_from_centroid']
        angle_data = sequence_with_polar[f'{hand}_hand_angle_from_centroid']

        try:
            ring_labels = pd.qcut(radius_data, q=4, labels=[1, 2, 3, 4])
            for ring in [1, 2, 3, 4]:
                count_in_ring = (ring_labels == ring).sum()
                fraction_in_ring = (ring_labels == ring).mean()
                extra[f'{hand}_ring_{ring}_frame_count'] = count_in_ring
                extra[f'{hand}_ring_{ring}_frame_fraction'] = fraction_in_ring
        except Exception:
            for ring in [1, 2, 3, 4]:
                extra[f'{hand}_ring_{ring}_frame_count'] = len(radius_data) if ring == 1 else 0
                extra[f'{hand}_ring_{ring}_frame_fraction'] = 1.0 if ring == 1 else 0.0

        angle_degrees = np.degrees(angle_data)

        def classify_angle_to_sector(angle_deg):
            normalized_angle = angle_deg % 360
            if normalized_angle <= 45 or normalized_angle > 315:
                return 1
            elif 45 < normalized_angle <= 135:
                return 2
            elif 135 < normalized_angle <= 225:
                return 3
            else:
                return 4

        sector_labels = [classify_angle_to_sector(angle) for angle in angle_degrees]

        for sector in [1, 2, 3, 4]:
            count_in_sector = sum(1 for s in sector_labels if s == sector)
            fraction_in_sector = count_in_sector / len(sector_labels) if len(sector_labels) > 0 else 0.0
            extra[f'{hand}_sector_{sector}_frame_count'] = count_in_sector
            extra[f'{hand}_sector_{sector}_frame_fraction'] = fraction_in_sector

        ring_fractions = [extra[f'{hand}_ring_{r}_frame_fraction'] for r in [1, 2, 3, 4]]
        dominant_ring = ring_fractions.index(max(ring_fractions)) + 1
        extra[f'{hand}_dominant_ring'] = dominant_ring

        sector_fractions = [extra[f'{hand}_sector_{s}_frame_fraction'] for s in [1, 2, 3, 4]]
        dominant_sector = sector_fractions.index(max(sector_fractions)) + 1
        extra[f'{hand}_dominant_sector'] = dominant_sector

        ring_entropy = -sum(p * np.log(p + 1e-12) for p in ring_fractions if p > 0)
        extra[f'{hand}_ring_entropy'] = ring_entropy

        sector_entropy = -sum(p * np.log(p + 1e-12) for p in sector_fractions if p > 0)
        extra[f'{hand}_sector_entropy'] = sector_entropy

        extra[f'{hand}_spatial_entropy'] = (ring_entropy + sector_entropy) / 2.0

        weighted_ring_preference = sum(ring_fractions[i] * (i + 1) for i in range(4))
        extra[f'{hand}_radial_preference'] = weighted_ring_preference

        sectors_used = sum(1 for p in sector_fractions if p > 0.1)
        extra[f'{hand}_angular_spread'] = sectors_used

    return pd.Series(extra)


def process_pose_data(
    df: pd.DataFrame,
    window_size: int = 7,
    polyorder: int = 2,
    threshold: float = 0.5,
    fps: float = 30.0,
    round_decimals: int = 6,
    compute_cog: bool = True,
    bin_relative: bool = True,
    bin_method: str = 'quartiles',
    bins_right=None,
    bins_left=None,
    return_sequence_agg: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    if 'label' not in df.columns:
        raise ValueError("Input DataFrame 'df' must contain a 'label' column for class-aware features.")

    df_processed = (
        df.groupby('id', group_keys=False)
          .apply(lambda g: _process_single_sequence(g, window_size, polyorder, threshold, fps))
    )

    if compute_cog:
        df_processed = _compute_cog_each_hand(df_processed)
        df_processed = _compute_distance_to_cog(df_processed)
        if bin_relative:
            df_processed = _bin_relative_distances(
                df_processed,
                method=bin_method,
                bins_right=bins_right,
                bins_left=bins_left
            )

    numeric_cols_processed = df_processed.select_dtypes(include=[np.number]).columns
    df_processed[numeric_cols_processed] = df_processed[numeric_cols_processed].round(round_decimals)

    df_seq_agg = None
    if return_sequence_agg:
        df_seq_agg = (
            df_processed.groupby('id', group_keys=True)
                        .apply(_aggregate_sequence_features)
                        .reset_index(drop=True)
        )

        numeric_agg = df_seq_agg.select_dtypes(include=[np.number]).columns
        df_seq_agg[numeric_agg] = df_seq_agg[numeric_agg].round(round_decimals)

        if 'label' not in df_seq_agg.columns:
            raise ValueError("'label' column is missing from df_seq_agg after aggregation.")
        if not pd.api.types.is_integer_dtype(df_seq_agg['label']):
            try:
                df_seq_agg['label'] = df_seq_agg['label'].astype(int)
            except ValueError as e:
                raise ValueError(f"Could not convert 'label' column to integer: {e}.")

        required_cols = ['hands_total_energy_sum', 'right_hand_velocity_mean', 'left_hand_velocity_mean']
        for col in required_cols:
            if col not in df_seq_agg.columns:
                raise ValueError(f"Required column '{col}' not found in df_seq_agg.")

        max_energy   = df_seq_agg['hands_total_energy_sum'].max()
        max_rvel     = df_seq_agg['right_hand_velocity_mean'].max()
        max_lvel     = df_seq_agg['left_hand_velocity_mean'].max()
        eps = 1e-10

        df_seq_agg['movement_score'] = (
            (df_seq_agg['hands_total_energy_sum'] / (max_energy + eps)) +
            (df_seq_agg['right_hand_velocity_mean'] / (max_rvel + eps)) +
            (df_seq_agg['left_hand_velocity_mean'] / (max_lvel + eps))
        ) / 3.0

        df_seq_agg['movement_quartile'] = '1'
        unique_labels = sorted(df_seq_agg['label'].unique())

        for lbl in unique_labels:
            mask = df_seq_agg['label'] == lbl
            scores = df_seq_agg.loc[mask, 'movement_score']
            n_samples = len(scores)
            if n_samples < 4:
                df_seq_agg.loc[mask, 'movement_quartile'] = '1'
            else:
                try:
                    quart_labels = pd.qcut(scores.rank(method='first'), q=4, labels=['1', '2', '3', '4'], duplicates='drop')
                    df_seq_agg.loc[mask, 'movement_quartile'] = quart_labels.astype(str)
                except Exception:
                    df_seq_agg.loc[mask, 'movement_quartile'] = '1'

        df_seq_agg['right_hand_distance_quartile'] = pd.qcut(
            df_seq_agg['right_hand_total_distance'], q=4, labels=['1', '2', '3', '4']
        )
        df_seq_agg['left_hand_distance_quartile'] = pd.qcut(
            df_seq_agg['left_hand_total_distance'], q=4, labels=['1', '2', '3', '4']
        )
        df_seq_agg['both_hands_distance_quartile'] = pd.qcut(
            df_seq_agg['both_hands_total_distance'], q=4, labels=['1', '2', '3', '4']
        )

        df_extra = (
            df_processed.groupby('id', group_keys=True)
                        .apply(_build_extra_features)
                        .reset_index()
        )

        df_seq_agg = df_seq_agg.merge(df_extra, on='id', how='left')

        df_seq_agg['right_radius_max_norm'] = (
            df_seq_agg['right_radius_max'] - df_seq_agg['right_radius_max'].min()
        ) / (df_seq_agg['right_radius_max'].max() - df_seq_agg['right_radius_max'].min() + eps)
        df_seq_agg['left_radius_max_norm'] = (
            df_seq_agg['left_radius_max'] - df_seq_agg['left_radius_max'].min()
        ) / (df_seq_agg['left_radius_max'].max() - df_seq_agg['left_radius_max'].min() + eps)

        df_seq_agg['right_frac_radius_gt_3.5_norm'] = (
            df_seq_agg['right_frac_radius_gt_3.5'] - df_seq_agg['right_frac_radius_gt_3.5'].min()
        ) / (df_seq_agg['right_frac_radius_gt_3.5'].max() - df_seq_agg['right_frac_radius_gt_3.5'].min() + eps)
        df_seq_agg['left_frac_radius_gt_3.5_norm'] = (
            df_seq_agg['left_frac_radius_gt_3.5'] - df_seq_agg['left_frac_radius_gt_3.5'].min()
        ) / (df_seq_agg['left_frac_radius_gt_3.5'].max() - df_seq_agg['left_frac_radius_gt_3.5'].min() + eps)

        df_seq_agg['polar_extremity_score'] = (
            0.5 * (df_seq_agg['right_radius_max_norm'] + df_seq_agg['right_frac_radius_gt_3.5_norm']) +
            0.5 * (df_seq_agg['left_radius_max_norm']  + df_seq_agg['left_frac_radius_gt_3.5_norm'])
        ) / 2.0

        df_seq_agg['hands_sep_max_norm'] = (
            df_seq_agg['hands_sep_max'] - df_seq_agg['hands_sep_max'].min()
        ) / (df_seq_agg['hands_sep_max'].max() - df_seq_agg['hands_sep_max'].min() + eps)
        df_seq_agg['frac_hands_sep_gt_3.5_norm'] = (
            df_seq_agg['frac_hands_sep_gt_3.5'] - df_seq_agg['frac_hands_sep_gt_3.5'].min()
        ) / (df_seq_agg['frac_hands_sep_gt_3.5'].max() - df_seq_agg['frac_hands_sep_gt_3.5'].min() + eps)

        df_seq_agg['separation_extremity_score'] = (
            df_seq_agg['hands_sep_max_norm'] + df_seq_agg['frac_hands_sep_gt_3.5_norm']
        ) / 2.0

        for hand in ['right', 'left']:
            df_seq_agg[f'{hand}_cog_dist_max_norm'] = (
                df_seq_agg[f'{hand}_cog_dist_max'] - df_seq_agg[f'{hand}_cog_dist_max'].min()
            ) / (df_seq_agg[f'{hand}_cog_dist_max'].max() - df_seq_agg[f'{hand}_cog_dist_max'].min() + eps)
            df_seq_agg[f'{hand}_frac_far_cog_gt_3.0_norm'] = (
                df_seq_agg[f'{hand}_frac_far_cog_gt_3.0'] - df_seq_agg[f'{hand}_frac_far_cog_gt_3.0'].min()
            ) / (df_seq_agg[f'{hand}_frac_far_cog_gt_3.0'].max() - df_seq_agg[f'{hand}_frac_far_cog_gt_3.0'].min() + eps)

        df_seq_agg['cog_extremity_score'] = (
            0.5 * (df_seq_agg['right_cog_dist_max_norm'] + df_seq_agg['right_frac_far_cog_gt_3.0_norm']) +
            0.5 * (df_seq_agg['left_cog_dist_max_norm']  + df_seq_agg['left_frac_far_cog_gt_3.0_norm'])
        ) / 2.0

        for hand in ['right', 'left']:
            df_seq_agg[f'{hand}_vel_mean_norm'] = (
                df_seq_agg[f'{hand}_hand_velocity_mean'] - df_seq_agg[f'{hand}_hand_velocity_mean'].min()
            ) / (df_seq_agg[f'{hand}_hand_velocity_mean'].max() - df_seq_agg[f'{hand}_hand_velocity_mean'].min() + eps)
            df_seq_agg[f'{hand}_frac_frames_fast_norm'] = (
                df_seq_agg[f'{hand}_frac_frames_fast'] - df_seq_agg[f'{hand}_frac_frames_fast'].min()
            ) / (df_seq_agg[f'{hand}_frac_frames_fast'].max() - df_seq_agg[f'{hand}_frac_frames_fast'].min() + eps)

        df_seq_agg['velocity_intensity_score'] = (
            0.5 * (df_seq_agg['right_vel_mean_norm'] + df_seq_agg['right_frac_frames_fast_norm']) +
            0.5 * (df_seq_agg['left_vel_mean_norm']  + df_seq_agg['left_frac_frames_fast_norm'])
        ) / 2.0

        for hand in ['right', 'left']:
            df_seq_agg[f'{hand}_acc_mean_norm'] = (
                df_seq_agg[f'{hand}_hand_accel_mean'] - df_seq_agg[f'{hand}_hand_accel_mean'].min()
            ) / (df_seq_agg[f'{hand}_hand_accel_mean'].max() - df_seq_agg[f'{hand}_hand_accel_mean'].min() + eps)
            df_seq_agg[f'{hand}_acc_iqr_norm'] = (
                df_seq_agg[f'{hand}_hand_accel_iqr'] - df_seq_agg[f'{hand}_hand_accel_iqr'].min()
            ) / (df_seq_agg[f'{hand}_hand_accel_iqr'].max() - df_seq_agg[f'{hand}_hand_accel_iqr'].min() + eps)
            df_seq_agg[f'{hand}_acc_sign_changes_norm'] = (
                df_seq_agg[f'{hand}_hand_accel_sign_changes'] - df_seq_agg[f'{hand}_hand_accel_sign_changes'].min()
            ) / (df_seq_agg[f'{hand}_hand_accel_sign_changes'].max() - df_seq_agg[f'{hand}_hand_accel_sign_changes'].min() + eps)

        df_seq_agg['acceleration_intensity_score'] = (
            0.33 * (df_seq_agg['right_acc_mean_norm'] + df_seq_agg['right_acc_iqr_norm'] + df_seq_agg['right_acc_sign_changes_norm']) +
            0.33 * (df_seq_agg['left_acc_mean_norm']  + df_seq_agg['left_acc_iqr_norm']  + df_seq_agg['left_acc_sign_changes_norm'])
        ) / 2.0

        df_seq_agg['hands_total_energy_sum_norm'] = (
            df_seq_agg['hands_total_energy_sum'] - df_seq_agg['hands_total_energy_sum'].min()
        ) / (df_seq_agg['hands_total_energy_sum'].max() - df_seq_agg['hands_total_energy_sum'].min() + eps)
        df_seq_agg['hands_total_energy_max_norm'] = (
            df_seq_agg['hands_total_energy_max'] - df_seq_agg['hands_total_energy_max'].min()
        ) / (df_seq_agg['hands_total_energy_max'].max() - df_seq_agg['hands_total_energy_max'].min() + eps)

        df_seq_agg['energy_intensity_score'] = (
            df_seq_agg['hands_total_energy_sum_norm'] + df_seq_agg['hands_total_energy_max_norm']
        ) / 2.0

        df_seq_agg['coord_mean_norm'] = (
            df_seq_agg['hands_coordination_coef_mean'] - df_seq_agg['hands_coordination_coef_mean'].min()
        ) / (df_seq_agg['hands_coordination_coef_mean'].max() - df_seq_agg['hands_coordination_coef_mean'].min() + eps)
        df_seq_agg['coord_max_norm'] = (
            df_seq_agg['hands_coordination_coef_max'] - df_seq_agg['hands_coordination_coef_max'].min()
        ) / (df_seq_agg['hands_coordination_coef_max'].max() - df_seq_agg['hands_coordination_coef_max'].min() + eps)
        df_seq_agg['frac_high_coord_norm'] = (
            df_seq_agg['hands_frac_frames_high_coord'] - df_seq_agg['hands_frac_frames_high_coord'].min()
        ) / (df_seq_agg['hands_frac_frames_high_coord'].max() - df_seq_agg['hands_frac_frames_high_coord'].min() + eps)

        df_seq_agg['coordination_score'] = (
            df_seq_agg['coord_mean_norm'] +
            df_seq_agg['coord_max_norm'] +
            df_seq_agg['frac_high_coord_norm']
        ) / 3.0

        df_seq_agg['hands_vertical_symmetry_mean'] = df_processed.groupby('id')['hands_vertical_symmetry'].mean().values
        df_seq_agg['hands_horizontal_symmetry_mean'] = df_processed.groupby('id')['hands_horizontal_symmetry'].mean().values

        df_seq_agg['vertical_sym_norm'] = (
            df_seq_agg['hands_vertical_symmetry_mean'] - df_seq_agg['hands_vertical_symmetry_mean'].min()
        ) / (df_seq_agg['hands_vertical_symmetry_mean'].max() - df_seq_agg['hands_vertical_symmetry_mean'].min() + eps)
        df_seq_agg['horizontal_sym_norm'] = (
            df_seq_agg['hands_horizontal_symmetry_mean'] - df_seq_agg['hands_horizontal_symmetry_mean'].min()
        ) / (df_seq_agg['hands_horizontal_symmetry_mean'].max() - df_seq_agg['hands_horizontal_symmetry_mean'].min() + eps)

        df_seq_agg['symmetry_score'] = 1.0 - (
            0.5 * df_seq_agg['vertical_sym_norm'] + 0.5 * df_seq_agg['horizontal_sym_norm']
        )

        def zone_entropy(hand: str) -> np.ndarray:
            zone_fracs = df_seq_agg[[f'{hand}_zone_{z}_frac' for z in range(1,10)]].values
            ent = -np.sum(zone_fracs * np.log(zone_fracs + 1e-12), axis=1)
            return ent

        df_seq_agg['right_zone_entropy'] = zone_entropy('right')
        df_seq_agg['left_zone_entropy']  = zone_entropy('left')

        df_seq_agg['right_zone_ent_norm'] = (
            df_seq_agg['right_zone_entropy'] - df_seq_agg['right_zone_entropy'].min()
        ) / (df_seq_agg['right_zone_entropy'].max() - df_seq_agg['right_zone_entropy'].min() + eps)
        df_seq_agg['left_zone_ent_norm']  = (
            df_seq_agg['left_zone_entropy'] - df_seq_agg['left_zone_entropy'].min()
        ) / (df_seq_agg['left_zone_entropy'].max() - df_seq_agg['left_zone_entropy'].min() + eps)

        df_seq_agg['zone_diversity_score'] = (
            df_seq_agg['right_zone_ent_norm'] + df_seq_agg['left_zone_ent_norm']
        ) / 2.0

        def quad_entropy(hand: str) -> np.ndarray:
            quad_fracs = df_seq_agg[[f'{hand}_quad_{q}_frac' for q in [1, 2, 3, 4]]].values
            ent = -np.sum(quad_fracs * np.log(quad_fracs + 1e-12), axis=1)
            return ent

        df_seq_agg['right_quad_entropy'] = quad_entropy('right')
        df_seq_agg['left_quad_entropy']  = quad_entropy('left')

        df_seq_agg['right_quad_ent_norm'] = (
            df_seq_agg['right_quad_entropy'] - df_seq_agg['right_quad_entropy'].min()
        ) / (df_seq_agg['right_quad_entropy'].max() - df_seq_agg['right_quad_entropy'].min() + eps)
        df_seq_agg['left_quad_ent_norm']  = (
            df_seq_agg['left_quad_entropy'] - df_seq_agg['left_quad_entropy'].min()
        ) / (df_seq_agg['left_quad_entropy'].max() - df_seq_agg['left_quad_entropy'].min() + eps)

        df_seq_agg['quadrant_diversity_score'] = (
            df_seq_agg['right_quad_ent_norm'] + df_seq_agg['left_quad_ent_norm']
        ) / 2.0

        for hand in ['right', 'left']:
            df_seq_agg[f'{hand}_path_ratio_norm'] = (
                df_seq_agg[f'{hand}_path_ratio'] - df_seq_agg[f'{hand}_path_ratio'].min()
            ) / (df_seq_agg[f'{hand}_path_ratio'].max() - df_seq_agg[f'{hand}_path_ratio'].min() + eps)

            df_seq_agg[f'{hand}_hull_area_norm'] = (
                df_seq_agg[f'{hand}_hull_area'] - df_seq_agg[f'{hand}_hull_area'].min()
            ) / (df_seq_agg[f'{hand}_hull_area'].max() - df_seq_agg[f'{hand}_hull_area'].min() + eps)

            df_seq_agg[f'{hand}_ellipse_area_norm'] = (
                df_seq_agg[f'{hand}_ellipse_area'] - df_seq_agg[f'{hand}_ellipse_area'].min()
            ) / (df_seq_agg[f'{hand}_ellipse_area'].max() - df_seq_agg[f'{hand}_ellipse_area'].min() + eps)

            df_seq_agg[f'{hand}_vel_sign_total'] = (
                df_seq_agg[f'{hand}_vel_sign_changes_x'] + df_seq_agg[f'{hand}_vel_sign_changes_y']
            )
            df_seq_agg[f'{hand}_vel_sign_total_norm'] = (
                df_seq_agg[f'{hand}_vel_sign_total'] - df_seq_agg[f'{hand}_vel_sign_total'].min()
            ) / (df_seq_agg[f'{hand}_vel_sign_total'].max() - df_seq_agg[f'{hand}_vel_sign_total'].min() + eps)

            df_seq_agg[f'{hand}_traj_complex'] = (
                0.25 * df_seq_agg[f'{hand}_path_ratio_norm'] +
                0.25 * df_seq_agg[f'{hand}_hull_area_norm'] +
                0.25 * df_seq_agg[f'{hand}_ellipse_area_norm'] +
                0.25 * df_seq_agg[f'{hand}_vel_sign_total_norm']
            )

        df_seq_agg['trajectory_complexity_score'] = (
            df_seq_agg['right_traj_complex'] + df_seq_agg['left_traj_complex']
        ) / 2.0

        for hand in ['right', 'left']:
            df_seq_agg[f'{hand}_first_frame_radius_gt_3.5_norm'] = 1.0 - (
                (df_seq_agg[f'{hand}_first_frame_radius_gt_3.5'] - df_seq_agg[f'{hand}_first_frame_radius_gt_3.5'].min())
                / (df_seq_agg[f'{hand}_first_frame_radius_gt_3.5'].max() - df_seq_agg[f'{hand}_first_frame_radius_gt_3.5'].min() + eps)
            )

        df_seq_agg['time_to_peak_sep'] = df_seq_agg['right_hand_time_to_peak_vel']

        df_seq_agg['time_to_peak_sep_norm'] = 1.0 - (
            (df_seq_agg['time_to_peak_sep'] - df_seq_agg['time_to_peak_sep'].min())
            / (df_seq_agg['time_to_peak_sep'].max() - df_seq_agg['time_to_peak_sep'].min() + eps)
        )

        df_seq_agg['time_to_extreme_score'] = (
            0.5 * df_seq_agg['right_first_frame_radius_gt_3.5_norm'] +
            0.5 * df_seq_agg['left_first_frame_radius_gt_3.5_norm'] +
            df_seq_agg['time_to_peak_sep_norm']
        ) / 3.0

        def early_late_mean_vel(seq: pd.DataFrame, hand: str) -> tuple[float, float]:
            n = len(seq)
            if n < 2:
                return np.nan, np.nan
            first_idx = int(np.floor(0.1 * n))
            last_idx_start = int(np.ceil(0.9 * n))
            early_mean = seq[f'{hand}_hand_velocity_total'].iloc[:first_idx].mean() if first_idx >= 1 else seq[f'{hand}_hand_velocity_total'].iloc[:1].mean()
            late_mean  = seq[f'{hand}_hand_velocity_total'].iloc[last_idx_start:].mean() if last_idx_start < n else seq[f'{hand}_hand_velocity_total'].iloc[-1:].mean()
            return early_mean, late_mean

        right_early,  right_late  = early_late_mean_vel(df_processed, 'right')
        left_early,   left_late   = early_late_mean_vel(df_processed, 'left')
        all_early_vals = df_seq_agg[['right_early', 'left_early']] = \
            pd.DataFrame({'right_early': [right_early], 'left_early': [left_early]})
        all_late_vals  = df_seq_agg[['right_late',  'left_late']]  = \
            pd.DataFrame({'right_late':  [right_late],  'left_late':  [left_late]})

        df_seq_agg['right_early_norm'] = (
            df_seq_agg['right_early'] - df_seq_agg['right_early'].min()
        ) / (df_seq_agg['right_early'].max() - df_seq_agg['right_early'].min() + eps)
        df_seq_agg['left_early_norm'] = (
            df_seq_agg['left_early'] - df_seq_agg['left_early'].min()
        ) / (df_seq_agg['left_early'].max() - df_seq_agg['left_early'].min() + eps)
        df_seq_agg['right_late_norm'] = (
            df_seq_agg['right_late'] - df_seq_agg['right_late'].min()
        ) / (df_seq_agg['right_late'].max() - df_seq_agg['right_late'].min() + eps)
        df_seq_agg['left_late_norm'] = (
            df_seq_agg['left_late'] - df_seq_agg['left_late'].min()
        ) / (df_seq_agg['left_late'].max() - df_seq_agg['left_late'].min() + eps)

        df_seq_agg['velocity_trend_score'] = (
            0.25 * (df_seq_agg['right_early_norm'] + df_seq_agg['right_late_norm']) +
            0.25 * (df_seq_agg['left_early_norm']  + df_seq_agg['left_late_norm'])
        )

        df_seq_agg['polar_ext_norm'] = (
            df_seq_agg['polar_extremity_score'] - df_seq_agg['polar_extremity_score'].min()
        ) / (df_seq_agg['polar_extremity_score'].max() - df_seq_agg['polar_extremity_score'].min() + eps)
        df_seq_agg['sep_ext_norm'] = (
            df_seq_agg['separation_extremity_score'] - df_seq_agg['separation_extremity_score'].min()
        ) / (df_seq_agg['separation_extremity_score'].max() - df_seq_agg['separation_extremity_score'].min() + eps)
        df_seq_agg['vel_int_norm'] = (
            df_seq_agg['velocity_intensity_score'] - df_seq_agg['velocity_intensity_score'].min()
        ) / (df_seq_agg['velocity_intensity_score'].max() - df_seq_agg['velocity_intensity_score'].min() + eps)
        df_seq_agg['acc_int_norm'] = (
            df_seq_agg['acceleration_intensity_score'] - df_seq_agg['acceleration_intensity_score'].min()
        ) / (df_seq_agg['acceleration_intensity_score'].max() - df_seq_agg['acceleration_intensity_score'].min() + eps)
        df_seq_agg['energy_int_norm'] = (
            df_seq_agg['energy_intensity_score'] - df_seq_agg['energy_intensity_score'].min()
        ) / (df_seq_agg['energy_intensity_score'].max() - df_seq_agg['energy_intensity_score'].min() + eps)

        df_seq_agg['global_ext_intensity_score'] = (
            df_seq_agg['polar_ext_norm'] +
            df_seq_agg['sep_ext_norm'] +
            df_seq_agg['vel_int_norm'] +
            df_seq_agg['acc_int_norm'] +
            df_seq_agg['energy_int_norm']
        ) / 5.0

        df_seq_agg['traj_comp_norm'] = (
            df_seq_agg['trajectory_complexity_score'] - df_seq_agg['trajectory_complexity_score'].min()
        ) / (df_seq_agg['trajectory_complexity_score'].max() - df_seq_agg['trajectory_complexity_score'].min() + eps)
        df_seq_agg['zone_div_norm'] = (
            df_seq_agg['zone_diversity_score'] - df_seq_agg['zone_diversity_score'].min()
        ) / (df_seq_agg['zone_diversity_score'].max() - df_seq_agg['zone_diversity_score'].min() + eps)
        df_seq_agg['coord_norm'] = (
            df_seq_agg['coordination_score'] - df_seq_agg['coordination_score'].min()
        ) / (df_seq_agg['coordination_score'].max() - df_seq_agg['coordination_score'].min() + eps)
        df_seq_agg['sym_norm'] = (
            df_seq_agg['symmetry_score'] - df_seq_agg['symmetry_score'].min()
        ) / (df_seq_agg['symmetry_score'].max() - df_seq_agg['symmetry_score'].min() + eps)

        df_seq_agg['global_complexity_coordination_score'] = (
            df_seq_agg['traj_comp_norm'] +
            df_seq_agg['zone_div_norm'] +
            df_seq_agg['coord_norm'] +
            df_seq_agg['sym_norm']
        ) / 4.0

        df_seq_agg['frac_frames_near_center'] = (
            df_extra['right_frac_radius_gt_3.5'].apply(lambda x: 1 - x) +
            df_extra['left_frac_radius_gt_3.5'].apply(lambda x: 1 - x)
        ) / 2.0

        df_seq_agg['near_center_norm'] = (
            df_seq_agg['frac_frames_near_center'] - df_seq_agg['frac_frames_near_center'].min()
        ) / (df_seq_agg['frac_frames_near_center'].max() - df_seq_agg['frac_frames_near_center'].min() + eps)

        df_seq_agg['time_ext_norm'] = (
            df_seq_agg['time_to_extreme_score'] - df_seq_agg['time_to_extreme_score'].min()
        ) / (df_seq_agg['time_to_extreme_score'].max() - df_seq_agg['time_to_extreme_score'].min() + eps)
        df_seq_agg['vel_trend_norm'] = (
            df_seq_agg['velocity_trend_score'] - df_seq_agg['velocity_trend_score'].min()
        ) / (df_seq_agg['velocity_trend_score'].max() - df_seq_agg['velocity_trend_score'].min() + eps)

        df_seq_agg['global_temporal_dynamics_score'] = (
            df_seq_agg['time_ext_norm'] +
            df_seq_agg['vel_trend_norm'] +
            df_seq_agg['near_center_norm']
        ) / 3.0

    return df_processed, df_seq_agg
