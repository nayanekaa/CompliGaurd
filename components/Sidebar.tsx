import React from 'react';
import { LayoutDashboard, MessageSquareText, FileText, ShieldAlert, Settings, LogOut } from 'lucide-react';
import { AppView } from '../types';

interface SidebarProps {
  currentView: AppView;
  onNavigate: (view: AppView) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({ currentView, onNavigate }) => {
  const navItems = [
    { id: AppView.DASHBOARD, label: 'Dashboard', icon: LayoutDashboard },
    { id: AppView.CHAT, label: 'Compliance Assistant', icon: MessageSquareText },
    { id: AppView.POLICIES, label: 'Policy Manager', icon: FileText },
    { id: AppView.MAPPING, label: 'Regulatory Map', icon: ShieldAlert },
  ];

  return (
    <aside className="w-64 bg-slate-900 text-slate-300 flex flex-col h-full border-r border-slate-800 shadow-xl">
      <div className="p-6 flex items-center gap-3 border-b border-slate-800">
        <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center">
          <ShieldAlert className="text-white w-5 h-5" />
        </div>
        <h1 className="text-white font-bold text-lg tracking-tight">CompliGuard</h1>
      </div>

      <nav className="flex-1 px-4 py-6 space-y-2">
        {navItems.map((item) => (
          <button
            key={item.id}
            onClick={() => onNavigate(item.id)}
            className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-colors duration-200
              ${currentView === item.id 
                ? 'bg-blue-600 text-white shadow-lg shadow-blue-900/20' 
                : 'hover:bg-slate-800 hover:text-white'
              }`}
          >
            <item.icon className="w-5 h-5" />
            {item.label}
          </button>
        ))}
      </nav>

      <div className="p-4 border-t border-slate-800 space-y-2">
        <button className="w-full flex items-center gap-3 px-4 py-2 text-sm font-medium hover:text-white transition-colors">
          <Settings className="w-4 h-4" />
          Settings
        </button>
        <button className="w-full flex items-center gap-3 px-4 py-2 text-sm font-medium hover:text-white transition-colors text-red-400 hover:text-red-300">
          <LogOut className="w-4 h-4" />
          Sign Out
        </button>
      </div>
      
      <div className="p-4 text-xs text-slate-600 text-center">
        v2.4.0 Enterprise Build
      </div>
    </aside>
  );
};
