import streamlit as st
import streamlit.components.v1 as components
import os
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import pandas as pd
from datetime import datetime

# Configuration
DATA_AREA = "/Users/angerami/Google Drive/My Drive/Data/transformer-analysis"
EXCLUDED_DIRS = ['src']

# Model and revision definitions
PYTHIA_REVISIONS = [
    "step0",
    "step1",
    "step2",
    "step4",
    "step8",
    "step16",
    "step32",
    "step64",
    "step128",
    "step256",
    "step512",
] + [f"step{step}" for step in range(1000, 144000, 1000)]

PYTHIA_MODELS = [
    'pythia-70m-deduped',
    'pythia-160m-deduped',
    'pythia-410m-deduped',
    'pythia-1b-deduped',
    'pythia-1.4b-deduped',
    'pythia-2.8b-deduped',
    'pythia-6.9b-deduped',
    'pythia-12b-deduped'
]


def format_size(size_bytes: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if abs(size_bytes) < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def get_dir_size_local(path: str) -> int:
    """Get local/apparent size of directory (what's actually on disk locally)."""
    if not os.path.exists(path):
        return 0
    total = 0
    try:
        for entry in os.scandir(path):
            if entry.is_file(follow_symlinks=False):
                # Use st_blocks * 512 for actual disk usage (handles sparse files/cloud placeholders)
                try:
                    stat = entry.stat()
                    total += stat.st_blocks * 512 if hasattr(stat, 'st_blocks') else stat.st_size
                except (OSError, AttributeError):
                    total += entry.stat().st_size
            elif entry.is_dir(follow_symlinks=False):
                total += get_dir_size_local(entry.path)
    except PermissionError:
        pass
    return total


def get_dir_size_logical(path: str) -> int:
    """Get logical/full size of directory (includes cloud-only files)."""
    if not os.path.exists(path):
        return 0
    total = 0
    try:
        for entry in os.scandir(path):
            if entry.is_file(follow_symlinks=False):
                total += entry.stat().st_size
            elif entry.is_dir(follow_symlinks=False):
                total += get_dir_size_logical(entry.path)
    except PermissionError:
        pass
    return total


def get_available_campaigns(data_dir: str) -> List[str]:
    """Get list of campaign directories, excluding hardcoded exclusions."""
    if not os.path.exists(data_dir):
        return []
    
    campaigns = []
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path) and item not in EXCLUDED_DIRS:
            campaigns.append(item)
    
    return sorted(campaigns)


def parse_directory_name(dirname: str) -> Tuple[str, str]:
    """
    Parse directory name into model_name and revision.
    Returns (model_name, revision) or (model_name, 'merged') for all_checkpoints dirs.
    """
    if dirname.endswith('_all_checkpoints'):
        model_name = dirname.replace('_all_checkpoints', '')
        return (model_name, 'merged')
    
    # Find the last underscore to split model name from revision
    parts = dirname.rsplit('_', 1)
    if len(parts) == 2:
        return (parts[0], parts[1])
    
    return (dirname, 'unknown')


def analyze_campaign(campaign_path: str) -> Dict[str, Dict]:
    """
    Analyze a campaign directory and return completion status for each model.
    
    Returns:
        Dict with structure:
        {
            'pythia-70m-deduped': {
                'completed': ['step0', 'step1', ...],
                'total': 154,
                'merged': True/False,
                'steps_size_local': int,
                'steps_size_gdrive': int,
                'merged_size_local': int,
                'merged_size_gdrive': int,
            },
            ...
        }
    """
    results = {}
    
    # Initialize results for all models
    for model in PYTHIA_MODELS:
        results[model] = {
            'completed': set(),
            'total': len(PYTHIA_REVISIONS),
            'merged': False,
            'steps_size_local': 0,
            'steps_size_gdrive': 0,
            'merged_size_local': 0,
            'merged_size_gdrive': 0,
        }
    
    # Check what actually exists
    if not os.path.exists(campaign_path):
        return results
    
    for item in os.listdir(campaign_path):
        item_path = os.path.join(campaign_path, item)
        if os.path.isdir(item_path):
            model_name, revision = parse_directory_name(item)
            
            if model_name in results:
                if revision == 'merged':
                    results[model_name]['merged'] = True
                    results[model_name]['merged_size_local'] = get_dir_size_local(item_path)
                    results[model_name]['merged_size_gdrive'] = get_dir_size_logical(item_path)
                elif revision in PYTHIA_REVISIONS:
                    results[model_name]['completed'].add(revision)
                    results[model_name]['steps_size_local'] += get_dir_size_local(item_path)
                    results[model_name]['steps_size_gdrive'] += get_dir_size_logical(item_path)
    
    return results


def load_performance_logs(campaign_path: str) -> Optional[pd.DataFrame]:
    """
    Load all JSON performance logs from the campaign's logs directory.
    
    Returns:
        DataFrame with performance metrics from all log files, or None if no logs found.
    """
    logs_dir = os.path.join(campaign_path, 'logs')
    
    if not os.path.exists(logs_dir):
        return None
    
    log_data = []
    
    for filename in os.listdir(logs_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(logs_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    
                    # Add filename info for tracking
                    data['log_file'] = filename
                    
                    # Extract model and revision from filename if possible
                    # Assuming format like: {model_name}_{revision}_log.json
                    name_parts = filename.replace('.json', '').split('_')
                    if len(name_parts) >= 2:
                        # Find where the step part starts
                        for i, part in enumerate(name_parts):
                            if part.startswith('step'):
                                data['model_name'] = '_'.join(name_parts[:i])
                                data['revision'] = '_'.join(name_parts[i:-1]) if name_parts[-1] == 'log' else '_'.join(name_parts[i:])
                                break
                    
                    log_data.append(data)
                    
            except (json.JSONDecodeError, Exception) as e:
                st.warning(f"Could not load {filename}: {e}")
                continue
    
    if not log_data:
        return None
    
    return pd.DataFrame(log_data)


def parse_performance_metrics(df: pd.DataFrame) -> Dict:
    """
    Extract and aggregate performance metrics from the logs DataFrame.
    """
    metrics = {
        'total_jobs': len(df),
        'total_duration': 0,
        'avg_duration': 0,
        'max_memory_delta_mb': 0,
        'avg_memory_delta_mb': 0,
        'max_memory_end_mb': 0,
        'avg_memory_end_mb': 0,
    }
    
    # Calculate total duration from all phases
    duration_cols = [
        'load_model_elapsed_sec',
        'configure_elapsed_sec',
        'loop_elapsed_sec',
        'finalize_elapsed_sec'
    ]
    
    available_duration_cols = [col for col in duration_cols if col in df.columns]
    if available_duration_cols:
        df['total_duration'] = df[available_duration_cols].sum(axis=1)
        metrics['total_duration'] = df['total_duration'].sum()
        metrics['avg_duration'] = df['total_duration'].mean()
    
    # Calculate total memory delta (peak memory usage)
    memory_delta_cols = [
        'load_model_memory_delta_mb',
        'configure_memory_delta_mb',
        'loop_memory_delta_mb',
        'finalize_memory_delta_mb'
    ]
    
    available_delta_cols = [col for col in memory_delta_cols if col in df.columns]
    if available_delta_cols:
        df['total_memory_delta'] = df[available_delta_cols].sum(axis=1)
        metrics['max_memory_delta_mb'] = df['total_memory_delta'].max()
        metrics['avg_memory_delta_mb'] = df['total_memory_delta'].mean()
    
    # Peak memory from end measurements
    memory_end_cols = [col for col in df.columns if col.endswith('_memory_end_mb')]
    if memory_end_cols:
        df['max_memory_end'] = df[memory_end_cols].max(axis=1)
        metrics['max_memory_end_mb'] = df['max_memory_end'].max()
        metrics['avg_memory_end_mb'] = df['max_memory_end'].mean()
    
    return metrics


def render_progress_bar(completed: int, total: int) -> str:
    """Create a simple text-based progress bar."""
    percentage = (completed / total * 100) if total > 0 else 0
    filled = int(percentage / 5)  # 20 blocks for 100%
    bar = '█' * filled + '░' * (20 - filled)
    return f"{bar} {completed}/{total} ({percentage:.1f}%)"


def copy_button_with_code(code: str, key: str):
    """Display code block with a copy button."""
    st.code(code, language="bash")
    # JavaScript-based copy button - escape backticks in the code
    escaped_code = code.replace('`', '\\`').replace('$', '\\$')
    components.html(f"""
        <button onclick="navigator.clipboard.writeText(`{escaped_code}`).then(() => this.innerText = '✓ Copied!')" 
                style="padding: 4px 12px; cursor: pointer; border-radius: 4px; 
                       border: 1px solid #ccc; background: #f0f0f0; font-size: 12px;">
            📋 Copy
        </button>
    """, height=35)


def main():
    st.set_page_config(
        page_title="Colab Campaign Monitor",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("🔬 Transformer Analysis Campaign Monitor")
    st.markdown("Monitor progress of Pythia model checkpoint analysis")
    
    # Get available campaigns
    campaigns = get_available_campaigns(DATA_AREA)
    
    if not campaigns:
        st.error(f"No campaigns found in: {DATA_AREA}")
        st.info("Make sure the data directory exists and contains campaign subdirectories.")
        return
    
    # Campaign selector
    selected_campaign = st.selectbox(
        "Select Campaign",
        campaigns,
        help="Choose a campaign to view its progress"
    )
    
    if selected_campaign:
        campaign_path = os.path.join(DATA_AREA, selected_campaign)
        st.markdown(f"**Campaign Path:** `{campaign_path}`")
        
        # Analyze campaign
        results = analyze_campaign(campaign_path)
        
        # Calculate overall statistics
        total_jobs = sum(r['total'] for r in results.values())
        completed_jobs = sum(len(r['completed']) for r in results.values())
        merged_count = sum(1 for r in results.values() if r['merged'])
        
        # Calculate total sizes
        total_steps_local = sum(r['steps_size_local'] for r in results.values())
        total_steps_gdrive = sum(r['steps_size_gdrive'] for r in results.values())
        total_merged_local = sum(r['merged_size_local'] for r in results.values())
        total_merged_gdrive = sum(r['merged_size_gdrive'] for r in results.values())
        
        # Display overall stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Checkpoints", total_jobs)
        with col2:
            st.metric("Completed", completed_jobs)
        with col3:
            st.metric("Remaining", total_jobs - completed_jobs)
        with col4:
            st.metric("Merged Models", f"{merged_count}/{len(PYTHIA_MODELS)}")
        
        # Display size stats
        st.markdown("#### 💾 Storage Usage")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Steps (Local)", format_size(total_steps_local))
        with col2:
            st.metric("Steps (GDrive)", format_size(total_steps_gdrive))
        with col3:
            st.metric("Merged (Local)", format_size(total_merged_local))
        with col4:
            st.metric("Merged (GDrive)", format_size(total_merged_gdrive))
        
        st.markdown("---")
        
        # Display per-model progress
        st.subheader("Model-by-Model Progress")
        
        for model in PYTHIA_MODELS:
            data = results[model]
            completed_count = len(data['completed'])
            total_count = data['total']
            percentage = (completed_count / total_count * 100) if total_count > 0 else 0
            
            with st.container():
                col1, col2, col3 = st.columns([3, 1, 2])
                
                with col1:
                    st.markdown(f"**{model}**")
                    st.progress(percentage / 100)
                    st.caption(render_progress_bar(completed_count, total_count))
                
                with col2:
                    # Merged status
                    merged_icon = "✅" if data['merged'] else "❌"
                    st.markdown(f"**Merged:** {merged_icon}")
                    
                    # Show missing count if any
                    if completed_count < total_count:
                        missing = total_count - completed_count
                        st.markdown(f"*Missing: {missing}*")
                
                with col3:
                    # Size information
                    st.markdown("**Storage:**")
                    if data['steps_size_gdrive'] > 0:
                        local_pct = (data['steps_size_local'] / data['steps_size_gdrive'] * 100) if data['steps_size_gdrive'] > 0 else 0
                        st.caption(f"Steps: {format_size(data['steps_size_local'])} / {format_size(data['steps_size_gdrive'])} ({local_pct:.0f}% local)")
                    else:
                        st.caption("Steps: —")
                    
                    if data['merged']:
                        local_pct = (data['merged_size_local'] / data['merged_size_gdrive'] * 100) if data['merged_size_gdrive'] > 0 else 0
                        st.caption(f"Merged: {format_size(data['merged_size_local'])} / {format_size(data['merged_size_gdrive'])} ({local_pct:.0f}% local)")
                    else:
                        st.caption("Merged: —")
                
                # Show delete command for merged models with completed steps
                if data['merged'] and completed_count > 0:
                    with st.expander(f"🗑️ Delete individual steps for {model} (free {format_size(data['steps_size_gdrive'])})"):
                        # Pattern matches model_step* but not model_all_checkpoints
                        delete_cmd = f'rm -rf "{campaign_path}/{model}_step"*'
                        copy_button_with_code(delete_cmd, key=f"delete_{model}")
                        st.caption(f"This will delete {completed_count} step directories, keeping the merged data.")
                
                st.markdown("")  # Spacing
        
        # Optional: Show detailed missing checkpoints
        with st.expander("🔍 View Missing Checkpoints Details"):
            for model in PYTHIA_MODELS:
                data = results[model]
                completed = data['completed']
                missing = [rev for rev in PYTHIA_REVISIONS if rev not in completed]
                
                if missing:
                    st.markdown(f"**{model}** - Missing {len(missing)} checkpoints:")
                    # Group missing checkpoints for readability
                    missing_str = ", ".join(missing[:10])
                    if len(missing) > 10:
                        missing_str += f" ... and {len(missing) - 10} more"
                    st.code(missing_str, language=None)
                else:
                    st.markdown(f"**{model}** - ✅ All checkpoints complete!")
        
        st.markdown("---")
        
        # Performance Monitoring Section
        st.subheader("📊 Performance & Resource Usage")
        
        perf_df = load_performance_logs(campaign_path)
        
        if perf_df is not None and len(perf_df) > 0:
            st.success(f"Loaded {len(perf_df)} performance logs")
            
            # Display available columns for debugging
            with st.expander("📋 Available Log Fields"):
                st.write(list(perf_df.columns))
            
            # Parse and display key metrics
            metrics = parse_performance_metrics(perf_df)
            
            # Display aggregate metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Avg Total Duration", f"{metrics['avg_duration']:.1f}s")
            with col2:
                st.metric("Peak Memory Delta", f"{metrics['max_memory_delta_mb']:.0f} MB")
            with col3:
                st.metric("Avg Memory Delta", f"{metrics['avg_memory_delta_mb']:.0f} MB")
            with col4:
                st.metric("Peak Memory End", f"{metrics['max_memory_end_mb']:.0f} MB")
            
            # Duration analysis by model
            if 'model_name' in perf_df.columns and 'total_duration' in perf_df.columns:
                st.markdown("#### ⏱️ Duration by Model")
                duration_by_model = perf_df.groupby('model_name')['total_duration'].agg(['mean', 'min', 'max', 'count'])
                duration_by_model.columns = ['Avg (s)', 'Min (s)', 'Max (s)', 'Jobs']
                st.dataframe(duration_by_model.style.format({
                    'Avg (s)': '{:.1f}',
                    'Min (s)': '{:.1f}',
                    'Max (s)': '{:.1f}',
                }), use_container_width=True)
                
                # Breakdown by phase
                st.markdown("##### Duration Breakdown by Phase")
                phase_cols = ['load_model_elapsed_sec', 'configure_elapsed_sec', 'loop_elapsed_sec', 'finalize_elapsed_sec']
                available_phases = [col for col in phase_cols if col in perf_df.columns]
                if available_phases and 'model_name' in perf_df.columns:
                    phase_means = perf_df.groupby('model_name')[available_phases].mean()
                    phase_means.columns = [col.replace('_elapsed_sec', '').replace('_', ' ').title() for col in phase_means.columns]
                    st.dataframe(phase_means.style.format('{:.2f}'), use_container_width=True)
            
            # Memory analysis by model
            if 'total_memory_delta' in perf_df.columns and 'model_name' in perf_df.columns:
                st.markdown("#### 💾 Memory Usage by Model")
                memory_by_model = perf_df.groupby('model_name')['total_memory_delta'].agg(['mean', 'min', 'max', 'count'])
                memory_by_model.columns = ['Avg Delta (MB)', 'Min Delta (MB)', 'Max Delta (MB)', 'Jobs']
                st.dataframe(memory_by_model.style.format({
                    'Avg Delta (MB)': '{:.0f}',
                    'Min Delta (MB)': '{:.0f}',
                    'Max Delta (MB)': '{:.0f}',
                }), use_container_width=True)
                
                # Breakdown by phase
                st.markdown("##### Memory Delta by Phase")
                mem_delta_cols = ['load_model_memory_delta_mb', 'configure_memory_delta_mb', 
                                  'loop_memory_delta_mb', 'finalize_memory_delta_mb']
                available_mem_deltas = [col for col in mem_delta_cols if col in perf_df.columns]
                if available_mem_deltas and 'model_name' in perf_df.columns:
                    mem_means = perf_df.groupby('model_name')[available_mem_deltas].mean()
                    mem_means.columns = [col.replace('_memory_delta_mb', '').replace('_', ' ').title() for col in mem_means.columns]
                    st.dataframe(mem_means.style.format('{:.0f}'), use_container_width=True)
            
            # Detailed log viewer
            with st.expander("🔬 View Raw Performance Logs"):
                # Allow filtering by model
                if 'model_name' in perf_df.columns:
                    models_in_logs = sorted(perf_df['model_name'].unique())
                    selected_model = st.selectbox("Filter by Model", ['All'] + models_in_logs)
                    
                    if selected_model != 'All':
                        filtered_df = perf_df[perf_df['model_name'] == selected_model]
                    else:
                        filtered_df = perf_df
                else:
                    filtered_df = perf_df
                
                st.dataframe(filtered_df, use_container_width=True)
        
        else:
            st.info("No performance logs found in the campaign's logs directory.")
            st.caption(f"Expected location: `{os.path.join(campaign_path, 'logs')}`")


if __name__ == "__main__":
    main()