/* eslint-disable no-unused-vars */

import Plot from 'react-plotly.js'; // Ensure react-plotly.js and plotly.js are installed

const EEG3DVisualization = ({ plotDataList, processedChannels }) => {
  if (!plotDataList || plotDataList.length === 0) {
    return (
      <div className="p-4 text-center text-slate-500">
        No spectral data available for 3D visualization.
      </div>
    );
  }

  const channelColors = {};
  let defaultPlotlyColors = [];
  try {
    if (window.Plotly && window.Plotly.Colors && window.Plotly.Colors.DEFAULT) {
        defaultPlotlyColors = window.Plotly.Colors.DEFAULT;
    } else {
        defaultPlotlyColors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
            '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
            '#bcbd22', '#17becf'
        ];
    }
  } catch (e) {
     console.warn("Plotly default colors not accessible, using fallback.");
     defaultPlotlyColors = [ '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf' ];
  }


  if (processedChannels && Array.isArray(processedChannels) && processedChannels.length > 0) {
    processedChannels.forEach((channel, i) => {
      channelColors[channel] = defaultPlotlyColors[i % defaultPlotlyColors.length];
    });
  } else { // Fallback if processedChannels is not available or empty
    // eslint-disable-next-line no-unused-vars
    plotDataList.forEach((plot, i) => {
        if (!channelColors[plot.channel_name]) { // Assign only if not already assigned
            channelColors[plot.channel_name] = defaultPlotlyColors[Object.keys(channelColors).length % defaultPlotlyColors.length];
        }
    });
  }


  const traces = plotDataList.map((plot) => ({
    x: plot.freqs,
    z: plot.psd,
    y: Array(plot.freqs.length).fill(`${plot.channel_name} - E${plot.epoch_number}`),
    type: 'scatter3d',
    mode: 'lines',
    name: `${plot.channel_name} (Epoch ${plot.epoch_number})`,
    line: {
      width: 4,
      color: channelColors[plot.channel_name] || defaultPlotlyColors[0]
    },
    hoverinfo: 'x+z+name' // Shows Freq (x), PSD (z), and Trace Name on hover
  }));

  const layout = {
    title: '3D EEG Power Spectral Density Analysis',
    autosize: true,
    height: 700,
    margin: { l: 10, r: 10, b: 50, t: 80 },
    scene: {
      xaxis: {
        title: 'Frequency (Hz)',
        backgroundcolor: "rgb(240, 240, 240)",
        gridcolor: "rgb(255, 255, 255)",
        showbackground: true,
        zerolinecolor: "rgb(255, 255, 255)",
        range: [0, Math.min(100, Math.max(...traces.flatMap(t => t.x)))] // Auto-adjust or set max like 100Hz
      },
      yaxis: {
        title: 'Channel & Epoch',
        type: 'category',
        backgroundcolor: "rgb(240, 240, 240)",
        gridcolor: "rgb(255, 255, 255)",
        showbackground: true,
        zerolinecolor: "rgb(255, 255, 255)",
        tickfont: {size: 10}
      },
      zaxis: {
        title: 'PSD (μV²/Hz)',
        type: 'log', // Using logarithmic scale for PSD
        backgroundcolor: "rgb(240, 240, 240)",
        gridcolor: "rgb(255, 255, 255)",
        showbackground: true,
        zerolinecolor: "rgb(255, 255, 255)",
      },
      camera: {
        eye: { x: 1.5, y: -2.2, z: 1.0 },
        up: { x: 0, y: 0, z: 1 }
      },
      aspectmode: 'cube'
    },
    legend: {
        orientation: 'v',
        x: 1.02,
        xanchor: 'left',
        y: 0.5,
        font: {size: 10}
    }
  };

  return (
    <div className="bg-white p-1 sm:p-2 md:p-4 rounded-lg shadow-inner border border-slate-200">
      <Plot
        data={traces}
        layout={layout}
        style={{ width: '100%', height: '100%' }}
        useResizeHandler={true}
        config={{ responsive: true, displaylogo: false, modeBarButtonsToRemove: ['toImage'] }}
      />
    </div>
  );
};

export default EEG3DVisualization;