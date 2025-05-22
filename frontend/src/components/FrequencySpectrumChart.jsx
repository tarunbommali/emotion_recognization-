/* eslint-disable react-hooks/rules-of-hooks */
// frontend/src/components/FrequencySpectrumChart.jsx
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  LogarithmicScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';
import annotationPlugin from 'chartjs-plugin-annotation';
import { useEffect, useRef, useMemo } from 'react';

ChartJS.register(
  CategoryScale,
  LinearScale,
  LogarithmicScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
  annotationPlugin
);

const FrequencySpectrumChart = ({ plotData }) => {
  const chartRef = useRef(null);

  const {
    epoch_number = 'N/A',
    channel_name = 'N/A',
    freqs = [],
    psd = [],
    bands = {}
  } = plotData || {};

  if (!plotData || freqs.length === 0 || psd.length === 0) {
    return <p className="text-center text-gray-500 dark:text-gray-400">No plot data available for this epoch.</p>;
  }

  const chartId = `spectrum-chart-${channel_name}-${epoch_number}`;

  const chartData = useMemo(() => ({
    labels: freqs.map(f => f.toFixed(2)),
    datasets: [
      {
        label: `PSD for ${channel_name}`,
        data: psd,
        borderColor: 'rgb(54, 162, 235)',
        backgroundColor: 'rgba(54, 162, 235, 0.2)',
        tension: 0.1,
        fill: 'origin',
        pointRadius: 1,
        borderWidth: 1.5,
      },
    ],
  }), [freqs, psd, channel_name]);


  const chartOptions = useMemo(() => {
    const bandColors = {
      delta: 'rgba(0, 0, 255, 0.1)',
      theta: 'rgba(0, 128, 0, 0.1)',
      alpha: 'rgba(255, 0, 0, 0.1)',
      beta:  'rgba(128, 0, 128, 0.1)',
      gamma: 'rgba(255, 165, 0, 0.1)'
    };

    const currentAnnotations = {};
    let yAxisMaxVal = 1;
    const positivePsdValues = psd.filter(p => p > 0);

    if (positivePsdValues.length > 0) {
        yAxisMaxVal = Math.max(...positivePsdValues) * 1.1; // Ensure a bit of space above max
    } else if (psd.length > 0) {
        const maxPsd = Math.max(...psd);
        yAxisMaxVal = maxPsd <= 0 ? 1 : maxPsd * 1.1; // If max is 0 or negative, default to 1
    }


    if (bands && Object.keys(bands).length > 0) {
      Object.keys(bands).forEach((bandName, index) => {
        const [lowFreq, highFreq] = bands[bandName];
        currentAnnotations[`band${index}`] = {
          type: 'box',
          xMin: lowFreq,
          xMax: highFreq,
          yMin: psd.filter(p => p > 0).length > 0 ? Math.min(...psd.filter(p => p > 0)) / 10 : 0.0001, // Small value for log scale
          yMax: yAxisMaxVal,
          backgroundColor: bandColors[bandName.toLowerCase()] || 'rgba(128, 128, 128, 0.1)',
          borderColor: 'transparent',
          borderWidth: 0,
          label: {
            content: bandName.charAt(0).toUpperCase() + bandName.slice(1),
            display: true,
            position: 'start',
            color: 'rgba(0,0,0,0.7)',
            font: { size: 9, weight: 'bold' },
            xAdjust: 5,
            yAdjust: -5,
          }
        };
      });
    }

    return {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      scales: {
        x: {
          title: { display: true, text: 'Frequency (Hz)', color: '#666', font: { size: 14 } },
          ticks: { color: '#666', maxTicksLimit: 15 },
          grid: { color: 'rgba(200, 200, 200, 0.2)' },
          max: 60, // Limit x-axis to a typical EEG range like 0-60 Hz
        },
        y: {
          type: 'logarithmic',
          title: { display: true, text: 'PSD (μV²/Hz)', color: '#666', font: { size: 14 } },
          ticks: {
            color: '#666',
            callback: function(value) {
                if (value === 0) return '0'; // Handle 0 explicitly for log scale if needed
                const k = 1000;
                const sizes = ['', 'K', 'M', 'G', 'T']; // For very large numbers, not typical for PSD
                const i = Math.floor(Math.log(Math.abs(value)) / Math.log(k)); // Use Math.abs for negative/zero
                if (Number.isInteger(Math.log10(Math.abs(value))) || value === 0.1 || value === 0.01 || value === 0.001) {
                    return value.toPrecision(1) + (sizes[i] || '');
                }
                return null;
            }
          },
          grid: { color: 'rgba(200, 200, 200, 0.2)' },
          min: psd.filter(p => p > 0).length > 0 ? Math.min(...psd.filter(p => p > 0)) / 10 : 0.0001,
          // max: yAxisMaxVal, // Can set max if needed, but auto-scaling usually works
        },
      },
      plugins: {
        legend: { position: 'top', labels: { color: '#333' } },
        title: { display: true, text: `Spectrum - Epoch ${epoch_number} (${channel_name})`, color: '#333', font: { size: 18, weight: 'bold' } },
        tooltip: { mode: 'index', intersect: false, },
        annotation: { annotations: currentAnnotations }
      },
      interaction: {
          mode: 'nearest', axis: 'x', intersect: false
      }
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [plotData, psd, freqs, bands, channel_name, epoch_number]);

  useEffect(() => {
    const chartInstance = chartRef.current;
    return () => {
      if (chartInstance) {
        chartInstance.destroy();
      }
    };
  }, []);

  useEffect(() => {
    if (chartRef.current) {
        chartRef.current.data = chartData;
        chartRef.current.options = chartOptions;
        chartRef.current.update();
    }
  }, [chartData, chartOptions]);


  return (
    <div className="bg-white p-4 md:p-6 rounded-lg shadow-lg h-96">
      <Line
        ref={chartRef}
        id={chartId}
        options={chartOptions}
        data={chartData}
      />
    </div>
  );
};

export default FrequencySpectrumChart;