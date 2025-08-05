import React, { useEffect, useState } from 'react';
import axios from 'axios';
import {
  LineChart, Line, XAxis, YAxis, Tooltip, Legend, CartesianGrid
} from 'recharts';

const ChartView = () => {
  const [prices, setPrices] = useState([]);
  const [events, setEvents] = useState([]);
  const [changePoints, setChangePoints] = useState([]);

  useEffect(() => {
    axios.get('/api/prices').then(res => setPrices(res.data));
    axios.get('/api/events').then(res => setEvents(res.data));
    axios.get('/api/change-points').then(res => setChangePoints(res.data));
  }, []);

  return (
    <div>
      <h2>Brent Oil Prices</h2>
      <LineChart width={1000} height={400} data={prices}>
        <XAxis dataKey="Date" />
        <YAxis />
        <Tooltip />
        <Legend />
        <CartesianGrid stroke="#ccc" />
        <Line type="monotone" dataKey="Price" stroke="#000" />
        {changePoints.map((cp, idx) => (
          <Line key={idx} dataKey="Price" stroke="red" strokeDasharray="5 5" dot={false} data={[{Date: cp.Date, Price: cp.Price}]} />
        ))}
      </LineChart>
    </div>
  );
};

export default ChartView;
