import React from 'react';
import { MOCK_REGULATIONS } from '../constants';
import { Shield, AlertTriangle, CheckCircle, ExternalLink, Activity } from 'lucide-react';

export const RegulatoryMap: React.FC = () => {
  return (
    <div className="flex flex-col h-full bg-slate-50">
        <div className="bg-white border-b border-slate-200 px-8 py-5">
            <h1 className="text-2xl font-bold text-slate-900">Regulatory Mapping</h1>
            <p className="text-sm text-slate-500 mt-1">Traceability matrix between external regulations (ISO, GDPR) and internal controls.</p>
        </div>

        <div className="flex-1 overflow-auto p-8">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
                 <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm flex items-center gap-4">
                    <div className="p-3 bg-emerald-100 rounded-lg text-emerald-600">
                        <Shield className="w-6 h-6" />
                    </div>
                    <div>
                        <div className="text-2xl font-bold text-slate-900">82%</div>
                        <div className="text-xs text-slate-500 font-medium uppercase tracking-wide">Coverage Score</div>
                    </div>
                 </div>
                  <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm flex items-center gap-4">
                    <div className="p-3 bg-red-100 rounded-lg text-red-600">
                        <AlertTriangle className="w-6 h-6" />
                    </div>
                    <div>
                        <div className="text-2xl font-bold text-slate-900">3</div>
                        <div className="text-xs text-slate-500 font-medium uppercase tracking-wide">Critical Gaps</div>
                    </div>
                 </div>
                 <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm flex items-center gap-4">
                    <div className="p-3 bg-blue-100 rounded-lg text-blue-600">
                        <Activity className="w-6 h-6" />
                    </div>
                    <div>
                        <div className="text-2xl font-bold text-slate-900">24</div>
                        <div className="text-xs text-slate-500 font-medium uppercase tracking-wide">Mapped Controls</div>
                    </div>
                 </div>
            </div>

            <div className="bg-white rounded-xl border border-slate-200 shadow-sm">
                <div className="px-6 py-4 border-b border-slate-200 flex justify-between items-center">
                    <h3 className="font-semibold text-slate-900">Regulation Requirements</h3>
                    <select className="bg-slate-50 border border-slate-300 rounded-md text-sm px-3 py-1.5 focus:outline-none focus:ring-2 focus:ring-blue-500">
                        <option>All Frameworks</option>
                        <option>ISO 27001</option>
                        <option>GDPR</option>
                        <option>SOC 2</option>
                    </select>
                </div>
                
                <div className="divide-y divide-slate-100">
                    {MOCK_REGULATIONS.map(reg => (
                        <div key={reg.id} className="p-6 hover:bg-slate-50 transition-colors">
                            <div className="flex items-start gap-4">
                                <div className={`mt-1 rounded-full p-1.5 shrink-0
                                    ${reg.status === 'Compliant' ? 'bg-emerald-100 text-emerald-600' :
                                      reg.status === 'Gap' ? 'bg-red-100 text-red-600' : 'bg-amber-100 text-amber-600'}`}>
                                    {reg.status === 'Compliant' ? <CheckCircle className="w-5 h-5" /> :
                                     reg.status === 'Gap' ? <AlertTriangle className="w-5 h-5" /> : <Activity className="w-5 h-5" />}
                                </div>
                                
                                <div className="flex-1">
                                    <div className="flex items-center justify-between mb-1">
                                        <div className="flex items-center gap-2">
                                            <span className="font-bold text-slate-900">{reg.name}</span>
                                            <span className="px-2 py-0.5 bg-slate-100 text-slate-600 rounded text-xs font-mono">{reg.section}</span>
                                        </div>
                                        <span className={`text-xs font-bold px-2 py-1 rounded-full border
                                            ${reg.status === 'Compliant' ? 'bg-emerald-50 text-emerald-700 border-emerald-200' :
                                              reg.status === 'Gap' ? 'bg-red-50 text-red-700 border-red-200' : 'bg-amber-50 text-amber-700 border-amber-200'}`}>
                                            {reg.status.toUpperCase()}
                                        </span>
                                    </div>
                                    
                                    <p className="text-slate-600 text-sm mb-3">{reg.description}</p>
                                    
                                    <div className="flex items-center gap-6 text-sm">
                                        <div className="flex items-center gap-2 text-slate-500">
                                            <span className="font-medium text-slate-700">Internal Control:</span>
                                            {reg.mappedControlId ? (
                                                <span className="font-mono text-blue-600 hover:underline cursor-pointer">{reg.mappedControlId}</span>
                                            ) : (
                                                <span className="text-red-500 italic">Not mapped</span>
                                            )}
                                        </div>
                                        {reg.evidenceCount > 0 && (
                                            <div className="flex items-center gap-1 text-slate-500">
                                                <Shield className="w-3.5 h-3.5" />
                                                <span>{reg.evidenceCount} evidence items</span>
                                            </div>
                                        )}
                                    </div>
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    </div>
  );
};
