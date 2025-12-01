import { PolicyDocument, Regulation } from './types';

// Mock Knowledge Base to simulate RAG context retrieval
export const MOCK_KNOWLEDGE_BASE = `
[SOURCE: Global_HR_Leave_Policy_v2.4.pdf | Page 3 | Section: Sick Leave]
Employees are entitled to 10 days of paid sick leave per calendar year. For any sick leave extending beyond 3 consecutive days, a medical certificate must be uploaded to the HR portal. For leave exceeding 7 days, the employee must contact the Leave Management Team (leave@company.com) to initiate a Short-Term Disability review.

[SOURCE: Remote_Work_Standard_2024.pdf | Page 4 | Section: Hybrid Model]
Employees are permitted to work remotely up to 3 days per week (Tue-Thu are core in-office days). Requests for full-time remote work must be approved by a VP-level executive and HR. Core working hours are 10:00 AM to 3:00 PM local time.

[SOURCE: InfoSec_Access_Control_ISO27001.docx | Page 12 | Section: Privileged Access]
Access to production databases is restricted to the DevOps engineering team (Group ID: devops-prod-access). All access must be logged via the PAM (Privileged Access Management) system. Access reviews must be conducted quarterly by the CISO. Temporary access for developers requires a ticket approved by the Engineering Manager.

[SOURCE: Travel_Expense_Policy_2024.pdf | Page 2 | Section: Meals & Entertainment]
The daily meal allowance for business travel is $75 per day. Alcohol is not reimbursable unless part of a client entertainment event, which requires pre-approval from a Director. Receipts are required for any expense over $25.

[SOURCE: Incident_Response_Playbook.pdf | Page 1 | Section: Severity Definition]
Severity 1 (Critical) incidents involve confirmed data breach, ransomware, or total system outage. SLA for initial response is 15 minutes. The Crisis Management Team must be activated immediately via PagerDuty.
`;

export const MOCK_POLICIES: PolicyDocument[] = [
  { 
    id: '1', 
    title: 'Global Remote Work Policy', 
    category: 'HR', 
    lastUpdated: '2024-03-15', 
    status: 'Active', 
    owner: 'Sarah Jenkins', 
    coverage: 100, 
    version: '2.4',
    nextReviewDate: '2025-03-15'
  },
  { 
    id: '2', 
    title: 'Data Access & Classification', 
    category: 'InfoSec', 
    lastUpdated: '2024-04-02', 
    status: 'Active', 
    owner: 'Mike Ross', 
    coverage: 95,
    version: '1.1',
    nextReviewDate: '2024-10-02'
  },
  { 
    id: '3', 
    title: 'Travel & Expense Standard', 
    category: 'Finance', 
    lastUpdated: '2023-11-20', 
    status: 'Review', 
    owner: 'Finance Ops', 
    coverage: 80,
    version: '3.0',
    nextReviewDate: '2023-11-20' // Overdue
  },
  { 
    id: '4', 
    title: 'Incident Response Plan', 
    category: 'InfoSec', 
    lastUpdated: '2024-01-10', 
    status: 'Active', 
    owner: 'SOC Team', 
    coverage: 100,
    version: '4.2',
    nextReviewDate: '2025-01-10'
  },
  { 
    id: '5', 
    title: 'AI Usage & Governance', 
    category: 'Legal', 
    lastUpdated: '2024-05-01', 
    status: 'Draft', 
    owner: 'Legal Counsel', 
    coverage: 40,
    version: '0.9',
    nextReviewDate: '2024-06-01'
  },
];

export const MOCK_REGULATIONS: Regulation[] = [
  { id: 'r1', name: 'ISO 27001', section: 'A.9.2.1', description: 'User registration and de-registration', mappedControlId: 'CTRL-052', status: 'Compliant', evidenceCount: 4 },
  { id: 'r2', name: 'ISO 27001', section: 'A.9.2.3', description: 'Management of privileged access rights', mappedControlId: 'CTRL-088', status: 'Compliant', evidenceCount: 12 },
  { id: 'r3', name: 'GDPR', section: 'Art. 33', description: 'Notification of a personal data breach', mappedControlId: 'CTRL-012', status: 'Partial', evidenceCount: 1 },
  { id: 'r4', name: 'SOC 2', section: 'CC6.1', description: 'Logical access security software', mappedControlId: 'CTRL-099', status: 'Compliant', evidenceCount: 8 },
  { id: 'r5', name: 'HIPAA', section: '164.312(a)(1)', description: 'Access Control', mappedControlId: null, status: 'Gap', evidenceCount: 0 },
  { id: 'r6', name: 'NIST CSF', section: 'PR.AC-1', description: 'Identities and credentials are managed', mappedControlId: 'CTRL-052', status: 'Compliant', evidenceCount: 6 },
];
