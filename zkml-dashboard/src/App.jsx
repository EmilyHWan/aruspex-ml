import { useState } from 'react';
import { Button, H1, HTMLTable } from '@blueprintjs/core';
import { Bar } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend } from 'chart.js';
import zkmlData from './data/zkmlData';
import "@blueprintjs/core/lib/css/blueprint.css";
import React from 'react';
import AuditChatbotWidget from './components/AuditChatbotWidget';

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

function App() {
  const [data, setData] = useState(zkmlData);

  const chartData = {
    labels: data.map(item => item.missionId),
    datasets: [
      {
        label: 'Proof Generation Time (seconds)',
        data: data.map(item => item.proofTime),
        backgroundColor: 'rgba(75, 192, 192, 0.6)',
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: { position: 'top' },
      title: { display: true, text: 'ZKML Proof Generation Times by Mission' },
    },
  };

  return (
    <div style={{ padding: '20px' }}>
      <H1>ZKML System Dashboard</H1>
      <HTMLTable bordered interactive striped>
        <thead>
          <tr>
            <th>Mission ID</th>
            <th>Drone ID</th>
            <th>Proof Time (s)</th>
            <th>Accuracy (%)</th>
            <th>Outcome</th>
          </tr>
        </thead>
        <tbody>
          {data.map(item => (
            <tr key={item.missionId}>
              <td>{item.missionId}</td>
              <td>{item.droneId}</td>
              <td>{item.proofTime}</td>
              <td>{item.accuracy}</td>
              <td>{item.outcome}</td>
            </tr>
          ))}
        </tbody>
      </HTMLTable>
      <div style={{ marginTop: '20px', height: '400px' }}>
        <Bar data={chartData} options={chartOptions} />
      </div>
      <Button intent="primary" text="Refresh Data" style={{ marginTop: '20px' }} />
    </div>
  );

  return (
    <div className="dashboard">
        <h1>ZKML Dashboard</h1>
        <AuditChatbotWidget />
    </div>
);

}

export default App;