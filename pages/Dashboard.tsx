import React from 'react';
import { PieChart, Pie, Cell, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts';
import { TrendingUp, TrendingDown, AlertCircle, CheckCircle } from 'lucide-react';

const COVERAGE_DATA = [
  { name: 'Covered', value: 72, color: '#10b981' }, // Emerald-500
  { name: 'Partial', value: 18, color: '#f59e0b' }, // Amber-500
  { name: 'Gap', value: 10, color: '#ef4444' },    // Red-500
];

const RISK_DATA = [
  { name: 'Week 1', high: 12, medium: 24 },
  { name: 'Week 2', high: 10, medium: 22 },
  { name: 'Week 3', high: 8, medium: 18 },
  { name: 'Week 4', high: 5, medium: 15 },
];

export const Dashboard: React.FC = () => {
  return (
    <div className="p-8 space-y-8 overflow-y-auto h-full pb-24">
      <div className="flex justify-between items-end">
        <div>
          <h2 className="text-2xl font-bold text-slate-900">Compliance Overview</h2>
          <p className="text-slate-500 mt-1">Real-time posture across all mapped regulations.</p>
        </div>
        <div className="flex gap-3">
            <button className="px-4 py-2 bg-white border border-slate-300 text-slate-700 font-medium rounded-lg shadow-sm hover:bg-slate-50">Export Report</button>
            <button className="px-4 py-2 bg-blue-600 text-white font-medium rounded-lg shadow-sm hover:bg-blue-700">Run Gap Analysis</button>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
          <div className="flex justify-between items-start mb-4">
            <div className="p-2 bg-green-100 rounded-lg">
                <CheckCircle className="w-5 h-5 text-green-600" />
            </div>
            <span className="text-green-600 text-xs font-semibold bg-green-50 px-2 py-1 rounded-full flex items-center gap-1">
                <TrendingUp className="w-3 h-3" /> +2.4%
            </span>
          </div>
          <div className="text-3xl font-bold text-slate-900">92%</div>
          <div className="text-sm text-slate-500 mt-1">Overall Policy Adherence</div>
        </div>

        <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
          <div className="flex justify-between items-start mb-4">
            <div className="p-2 bg-red-100 rounded-lg">
                <AlertCircle className="w-5 h-5 text-red-600" />
            </div>
            <span className="text-red-600 text-xs font-semibold bg-red-50 px-2 py-1 rounded-full flex items-center gap-1">
                <TrendingDown className="w-3 h-3" /> -1
            </span>
          </div>
          <div className="text-3xl font-bold text-slate-900">3</div>
          <div className="text-sm text-slate-500 mt-1">Open Critical Gaps</div>
        </div>

        <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
           <div className="flex justify-between items-start mb-4">
            <div className="p-2 bg-blue-100 rounded-lg">
                <TrendingUp className="w-5 h-5 text-blue-600" />
            </div>
          </div>
          <div className="text-3xl font-bold text-slate-900">14/15</div>
          <div className="text-sm text-slate-500 mt-1">Attestations Pending</div>
        </div>

        <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
           <div className="flex justify-between items-start mb-4">
            <div className="p-2 bg-amber-100 rounded-lg">
                <AlertCircle className="w-5 h-5 text-amber-600" />
            </div>
          </div>
          <div className="text-3xl font-bold text-slate-900">4</div>
          <div className="text-sm text-slate-500 mt-1">Policies Expiring Soon</div>
        </div>
      </div>

      {/* Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
          <h3 className="text-lg font-bold text-slate-900 mb-6">Control Coverage by Status</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={COVERAGE_DATA}
                  innerRadius={60}
                  outerRadius={80}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {COVERAGE_DATA.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div className="flex justify-center gap-6 mt-4">
            {COVERAGE_DATA.map((item) => (
                <div key={item.name} className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full" style={{ backgroundColor: item.color }}></div>
                    <span className="text-sm text-slate-600">{item.name}</span>
                </div>
            ))}
          </div>
        </div>

        <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
          <h3 className="text-lg font-bold text-slate-900 mb-6">Risk Remediation Burn-down</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={RISK_DATA}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
                <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{fill: '#64748b'}} />
                <YAxis axisLine={false} tickLine={false} tick={{fill: '#64748b'}} />
                <Tooltip cursor={{fill: '#f1f5f9'}} />
                <Bar dataKey="high" fill="#ef4444" radius={[4, 4, 0, 0]} stackId="a" name="High Risk" />
                <Bar dataKey="medium" fill="#f59e0b" radius={[4, 4, 0, 0]} stackId="a" name="Medium Risk" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
};