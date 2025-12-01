
import React, { useState, useRef } from 'react';
import { MOCK_POLICIES } from '../constants';
import { PolicyDocument } from '../types';
import { FileText, MoreVertical, Search, Filter, Plus, Clock, CheckCircle, AlertCircle, Upload, X, FileUp, Shield, Loader2, ScanLine, Database } from 'lucide-react';

export const Policies: React.FC = () => {
  const [policies, setPolicies] = useState<PolicyDocument[]>(MOCK_POLICIES);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  
  // Upload State
  const [uploadStep, setUploadStep] = useState<'idle' | 'uploading' | 'scanning' | 'indexing' | 'success'>('idle');
  const [newFile, setNewFile] = useState<File | null>(null);
  const [formData, setFormData] = useState({
    title: '',
    category: 'HR' as PolicyDocument['category'],
    owner: 'Me',
    version: '1.0'
  });
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setNewFile(file);
      // Auto-fill title from filename if empty
      if (!formData.title) {
        setFormData(prev => ({ ...prev, title: file.name.replace(/\.[^/.]+$/, "") }));
      }
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      setNewFile(file);
       if (!formData.title) {
        setFormData(prev => ({ ...prev, title: file.name.replace(/\.[^/.]+$/, "") }));
      }
    }
  };

  const startUploadProcess = () => {
    if (!newFile) return;

    setUploadStep('uploading');

    // Simulate the Enterprise RAG ingestion pipeline
    // 1. Upload
    setTimeout(() => {
        setUploadStep('scanning'); // 2. OCR / Text Extraction
        setTimeout(() => {
            setUploadStep('indexing'); // 3. Vector Embedding Generation
            setTimeout(() => {
                setUploadStep('success'); // 4. Complete
                
                // Add to list
                const newPolicy: PolicyDocument = {
                    id: Date.now().toString(),
                    title: formData.title,
                    category: formData.category,
                    lastUpdated: new Date().toISOString().split('T')[0],
                    status: 'Draft',
                    owner: formData.owner,
                    coverage: 0, // Initial coverage 0 until mapped
                    version: formData.version,
                    nextReviewDate: new Date(new Date().setFullYear(new Date().getFullYear() + 1)).toISOString().split('T')[0]
                };
                setPolicies(prev => [newPolicy, ...prev]);
                
                // Reset and close after delay
                setTimeout(() => {
                    setIsModalOpen(false);
                    setUploadStep('idle');
                    setNewFile(null);
                    setFormData({ title: '', category: 'HR', owner: 'Me', version: '1.0' });
                }, 1500);
            }, 2000);
        }, 2000);
    }, 1500);
  };

  const filteredPolicies = policies.filter(p => 
    p.title.toLowerCase().includes(searchQuery.toLowerCase()) || 
    p.owner.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="flex flex-col h-full bg-slate-50 relative">
      {/* Sub-header */}
      <div className="bg-white border-b border-slate-200 px-8 py-5 flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold text-slate-900">Policy Manager</h1>
          <p className="text-sm text-slate-500 mt-1">Manage lifecycle, versions, and approvals for internal SOPs.</p>
        </div>
        <button 
            onClick={() => setIsModalOpen(true)}
            className="flex items-center gap-2 bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg font-medium shadow-sm transition-colors"
        >
          <Plus className="w-4 h-4" />
          Upload New Policy
        </button>
      </div>

      {/* Toolbar */}
      <div className="px-8 py-4 flex items-center justify-between gap-4">
        <div className="relative flex-1 max-w-lg">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
          <input 
            type="text" 
            placeholder="Search policies..." 
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-10 pr-4 py-2 bg-white border border-slate-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none"
          />
        </div>
        <div className="flex items-center gap-2">
            <button className="flex items-center gap-2 px-3 py-2 bg-white border border-slate-300 rounded-lg text-sm text-slate-600 hover:bg-slate-50">
                <Filter className="w-4 h-4" /> Filter
            </button>
        </div>
      </div>

      {/* Table */}
      <div className="flex-1 overflow-auto px-8 pb-8">
        <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="bg-slate-50 border-b border-slate-200">
                <th className="px-6 py-4 text-xs font-semibold text-slate-500 uppercase tracking-wider">Document Name</th>
                <th className="px-6 py-4 text-xs font-semibold text-slate-500 uppercase tracking-wider">Category</th>
                <th className="px-6 py-4 text-xs font-semibold text-slate-500 uppercase tracking-wider">Status</th>
                <th className="px-6 py-4 text-xs font-semibold text-slate-500 uppercase tracking-wider">Owner</th>
                <th className="px-6 py-4 text-xs font-semibold text-slate-500 uppercase tracking-wider">Last Update</th>
                <th className="px-6 py-4 text-xs font-semibold text-slate-500 uppercase tracking-wider">Review Due</th>
                <th className="px-6 py-4 text-xs font-semibold text-slate-500 uppercase tracking-wider"></th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {filteredPolicies.map((policy) => (
                <tr key={policy.id} className="hover:bg-slate-50 transition-colors group">
                  <td className="px-6 py-4">
                    <div className="flex items-center gap-3">
                      <div className="w-8 h-8 rounded-lg bg-blue-50 flex items-center justify-center text-blue-600">
                        <FileText className="w-4 h-4" />
                      </div>
                      <div>
                        <div className="font-medium text-slate-900">{policy.title}</div>
                        <div className="text-xs text-slate-500">v{policy.version}</div>
                      </div>
                    </div>
                  </td>
                  <td className="px-6 py-4">
                    <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-slate-100 text-slate-800">
                      {policy.category}
                    </span>
                  </td>
                  <td className="px-6 py-4">
                    {policy.status === 'Active' && (
                         <span className="flex items-center gap-1.5 text-xs font-medium text-emerald-700 bg-emerald-50 px-2 py-1 rounded-full w-fit">
                            <CheckCircle className="w-3.5 h-3.5" /> Active
                         </span>
                    )}
                     {policy.status === 'Review' && (
                         <span className="flex items-center gap-1.5 text-xs font-medium text-amber-700 bg-amber-50 px-2 py-1 rounded-full w-fit">
                            <Clock className="w-3.5 h-3.5" /> In Review
                         </span>
                    )}
                     {policy.status === 'Draft' && (
                         <span className="flex items-center gap-1.5 text-xs font-medium text-slate-600 bg-slate-100 px-2 py-1 rounded-full w-fit">
                            <FileText className="w-3.5 h-3.5" /> Draft
                         </span>
                    )}
                  </td>
                  <td className="px-6 py-4">
                     <div className="flex items-center gap-2">
                        <div className="w-6 h-6 rounded-full bg-indigo-100 flex items-center justify-center text-xs font-bold text-indigo-700">
                            {policy.owner.charAt(0)}
                        </div>
                        <span className="text-sm text-slate-600">{policy.owner}</span>
                     </div>
                  </td>
                  <td className="px-6 py-4 text-sm text-slate-600">
                    {policy.lastUpdated}
                  </td>
                   <td className="px-6 py-4 text-sm text-slate-600">
                    {policy.nextReviewDate}
                  </td>
                  <td className="px-6 py-4 text-right">
                    <button className="text-slate-400 hover:text-slate-600 p-1">
                        <MoreVertical className="w-4 h-4" />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Upload Modal */}
      {isModalOpen && (
        <div className="fixed inset-0 bg-slate-900/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
            <div className="bg-white rounded-xl shadow-2xl w-full max-w-lg overflow-hidden animate-in fade-in zoom-in-95 duration-200">
                <div className="px-6 py-4 border-b border-slate-100 flex justify-between items-center bg-slate-50">
                    <h2 className="text-lg font-bold text-slate-800">Upload Policy Document</h2>
                    <button onClick={() => !uploadStep.startsWith('upload') && setIsModalOpen(false)} className="text-slate-400 hover:text-slate-600">
                        <X className="w-5 h-5" />
                    </button>
                </div>
                
                <div className="p-6">
                    {uploadStep === 'idle' ? (
                        <>
                            {/* Drop Zone */}
                            <div 
                                onDragOver={(e) => e.preventDefault()}
                                onDrop={handleDrop}
                                onClick={() => fileInputRef.current?.click()}
                                className="border-2 border-dashed border-slate-300 rounded-xl p-8 text-center hover:bg-slate-50 hover:border-blue-400 transition-colors cursor-pointer"
                            >
                                <input 
                                    type="file" 
                                    ref={fileInputRef}
                                    className="hidden" 
                                    accept=".pdf,.docx,.txt"
                                    onChange={handleFileSelect}
                                />
                                {newFile ? (
                                    <div className="flex flex-col items-center gap-3">
                                        <div className="w-12 h-12 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center">
                                            <FileText className="w-6 h-6" />
                                        </div>
                                        <div className="font-medium text-slate-900">{newFile.name}</div>
                                        <div className="text-xs text-slate-500">{(newFile.size / 1024).toFixed(1)} KB</div>
                                        <button 
                                            onClick={(e) => { e.stopPropagation(); setNewFile(null); setFormData(p => ({...p, title: ''})); }}
                                            className="text-xs text-red-500 hover:text-red-700 font-medium"
                                        >
                                            Remove
                                        </button>
                                    </div>
                                ) : (
                                    <div className="flex flex-col items-center gap-2">
                                        <div className="w-12 h-12 bg-slate-100 text-slate-400 rounded-full flex items-center justify-center mb-2">
                                            <Upload className="w-6 h-6" />
                                        </div>
                                        <div className="text-slate-900 font-medium">Click to upload or drag and drop</div>
                                        <div className="text-xs text-slate-500">PDF, DOCX, or TXT (Max 10MB)</div>
                                    </div>
                                )}
                            </div>

                            {/* Metadata Form */}
                            <div className="mt-6 space-y-4">
                                <div>
                                    <label className="block text-xs font-semibold text-slate-500 uppercase tracking-wide mb-1.5">Document Title</label>
                                    <input 
                                        type="text" 
                                        value={formData.title}
                                        onChange={(e) => setFormData({...formData, title: e.target.value})}
                                        className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 outline-none"
                                        placeholder="e.g. Remote Work Standard 2025"
                                    />
                                </div>
                                <div className="grid grid-cols-2 gap-4">
                                    <div>
                                        <label className="block text-xs font-semibold text-slate-500 uppercase tracking-wide mb-1.5">Category</label>
                                        <select 
                                            value={formData.category}
                                            onChange={(e) => setFormData({...formData, category: e.target.value as any})}
                                            className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 outline-none bg-white"
                                        >
                                            <option value="HR">HR</option>
                                            <option value="InfoSec">InfoSec</option>
                                            <option value="Legal">Legal</option>
                                            <option value="Finance">Finance</option>
                                            <option value="Ops">Ops</option>
                                        </select>
                                    </div>
                                    <div>
                                        <label className="block text-xs font-semibold text-slate-500 uppercase tracking-wide mb-1.5">Version</label>
                                        <input 
                                            type="text" 
                                            value={formData.version}
                                            onChange={(e) => setFormData({...formData, version: e.target.value})}
                                            className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 outline-none"
                                            placeholder="1.0"
                                        />
                                    </div>
                                </div>
                            </div>
                        </>
                    ) : (
                        <div className="py-8 flex flex-col items-center justify-center text-center">
                            {uploadStep === 'success' ? (
                                <>
                                    <div className="w-16 h-16 bg-emerald-100 text-emerald-600 rounded-full flex items-center justify-center mb-4 animate-in zoom-in duration-300">
                                        <CheckCircle className="w-8 h-8" />
                                    </div>
                                    <h3 className="text-lg font-bold text-slate-900">Policy Imported Successfully</h3>
                                    <p className="text-sm text-slate-500 mt-2">Document indexed and ready for compliance queries.</p>
                                </>
                            ) : (
                                <>
                                    <div className="w-16 h-16 bg-blue-50 text-blue-600 rounded-full flex items-center justify-center mb-4 relative">
                                        {/* Dynamic Icon based on step */}
                                        {uploadStep === 'scanning' ? (
                                             <ScanLine className="w-8 h-8 animate-pulse text-purple-600" />
                                        ) : uploadStep === 'indexing' ? (
                                             <Database className="w-8 h-8 animate-bounce text-amber-600" />
                                        ) : (
                                             <Loader2 className="w-8 h-8 animate-spin" />
                                        )}
                                        
                                        {uploadStep === 'uploading' && (
                                            <div className="absolute inset-0 border-4 border-blue-100 rounded-full"></div>
                                        )}
                                    </div>
                                    <h3 className="text-lg font-bold text-slate-900">
                                        {uploadStep === 'uploading' && 'Uploading Document...'}
                                        {uploadStep === 'scanning' && 'Scanning with OCR...'}
                                        {uploadStep === 'indexing' && 'Generating Vector Embeddings...'}
                                    </h3>
                                    <p className="text-sm text-slate-500 mt-2">
                                        {uploadStep === 'scanning' ? 'Extracting text and tables...' : 
                                         uploadStep === 'indexing' ? 'Storing in Vector Database...' : 
                                         'Securely transferring file...'}
                                    </p>
                                </>
                            )}
                        </div>
                    )}
                </div>

                <div className="px-6 py-4 border-t border-slate-100 bg-slate-50 flex justify-end gap-3">
                    {uploadStep === 'idle' && (
                        <>
                            <button 
                                onClick={() => setIsModalOpen(false)}
                                className="px-4 py-2 text-sm font-medium text-slate-600 hover:text-slate-800 hover:bg-slate-200 rounded-lg transition-colors"
                            >
                                Cancel
                            </button>
                            <button 
                                onClick={startUploadProcess}
                                disabled={!newFile || !formData.title}
                                className="px-4 py-2 text-sm font-medium bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed shadow-sm flex items-center gap-2"
                            >
                                <FileUp className="w-4 h-4" />
                                Upload & Index
                            </button>
                        </>
                    )}
                </div>
            </div>
        </div>
      )}
    </div>
  );
};
