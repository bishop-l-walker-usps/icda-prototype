import axios from 'axios';
import type { AxiosInstance } from 'axios';
import type {
  QueryRequest,
  QueryResponse,
  HealthStatus,
  CacheStats,
  AddressVerificationRequest,
  AddressVerificationResponse,
  SingleAddressVerificationResponse,
  FileQueryRequest,
  AutocompleteResult,
  SemanticSearchResult,
  KnowledgeDocument,
  KnowledgeStats,
  KnowledgeUploadResult,
  KnowledgeSearchResult,
  EnhancedValidationResponse,
  QuickValidationResponse,
} from '../types';

// Download result response type
export interface DownloadResult {
  success: boolean;
  query: string;
  total: number;
  data: Record<string, unknown>[];
  generated_at: string;
}

// API base URL - configurable via environment variable
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Create axios instance with defaults
const apiClient: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor - adds request timing
apiClient.interceptors.request.use(
  (config) => {
    // Add timestamp for latency tracking (using headers for compatibility)
    config.headers.set('X-Request-Start', Date.now().toString());
    return config;
  },
  (error) => {
    console.error('Request error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor - handles errors consistently
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    // Log error details for debugging
    if (error.response) {
      // Server responded with error status
      console.error(`API Error [${error.response.status}]:`, error.response.data);
    } else if (error.request) {
      // Request made but no response received
      console.error('Network error - no response received:', error.message);
    } else {
      // Error in request setup
      console.error('Request setup error:', error.message);
    }
    return Promise.reject(error);
  }
);

// Helper function to convert JSON data to CSV format
function convertToCSV(data: Record<string, unknown>[]): string {
  if (data.length === 0) return '';

  // Get all unique headers from all records
  const headers = new Set<string>();
  data.forEach((record) => {
    Object.keys(record).forEach((key) => headers.add(key));
  });
  const headerArray = Array.from(headers);

  // Create CSV rows
  const rows = data.map((record) =>
    headerArray
      .map((header) => {
        const value = record[header];
        if (value === null || value === undefined) return '';
        const stringValue = typeof value === 'object' ? JSON.stringify(value) : String(value);
        // Escape quotes and wrap in quotes if contains comma, newline, or quote
        if (stringValue.includes(',') || stringValue.includes('\n') || stringValue.includes('"')) {
          return `"${stringValue.replace(/"/g, '""')}"`;
        }
        return stringValue;
      })
      .join(',')
  );

  return [headerArray.join(','), ...rows].join('\n');
}

// API Functions
export const api = {
  // Query endpoint (JSON body)
  query: async (request: QueryRequest): Promise<QueryResponse> => {
    const response = await apiClient.post<QueryResponse>('/api/query', request);
    return response.data;
  },

  // Query with optional file upload (multipart form data)
  queryWithFile: async (request: FileQueryRequest): Promise<QueryResponse> => {
    const formData = new FormData();
    formData.append('query', request.query);
    if (request.file) {
      formData.append('file', request.file);
    }
    formData.append('bypass_cache', String(request.bypass_cache ?? false));
    if (request.session_id) {
      formData.append('session_id', request.session_id);
    }
    if (request.guardrails) {
      formData.append('pii', String(request.guardrails.pii));
      formData.append('financial', String(request.guardrails.financial));
      formData.append('credentials', String(request.guardrails.credentials));
      formData.append('offtopic', String(request.guardrails.offtopic));
    }
    const response = await apiClient.post<QueryResponse>('/api/query-with-file', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      timeout: 60000, // 1 minute for file uploads
    });
    return response.data;
  },

  // Health check endpoint
  health: async (): Promise<HealthStatus> => {
    const response = await apiClient.get<HealthStatus>('/api/health');
    return response.data;
  },

  // Cache stats endpoint
  cacheStats: async (): Promise<CacheStats> => {
    const response = await apiClient.get<CacheStats>('/api/cache/stats');
    return response.data;
  },

  // Clear cache endpoint
  clearCache: async (): Promise<void> => {
    await apiClient.delete('/api/cache');
  },

  // Session management
  newSession: async (): Promise<{ session_id: string }> => {
    const response = await apiClient.post<{ session_id: string }>('/api/session/new');
    return response.data;
  },

  deleteSession: async (sessionId: string): Promise<void> => {
    await apiClient.delete(`/api/session/${sessionId}`);
  },

  clearAllSessions: async (): Promise<{ count: number }> => {
    const response = await apiClient.delete<{ status: string; count: number }>('/api/sessions');
    return response.data;
  },

  // Address verification endpoint (for file upload)
  verifyAddresses: async (
    request: AddressVerificationRequest
  ): Promise<AddressVerificationResponse> => {
    const response = await apiClient.post<AddressVerificationResponse>(
      '/api/verify-addresses',
      request
    );
    return response.data;
  },

  // Bulk address upload endpoint
  uploadAddressFile: async (
    file: File,
    saveToPipeline: boolean
  ): Promise<AddressVerificationResponse> => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('save_to_pipeline', String(saveToPipeline));

    const response = await apiClient.post<AddressVerificationResponse>(
      '/api/upload-addresses',
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 120000, // 2 minutes for large files
      }
    );
    return response.data;
  },

  // Autocomplete endpoints for smart address suggestions
  autocomplete: async (
    field: 'address' | 'name' | 'city',
    query: string,
    limit: number = 10,
    fuzzy: boolean = false
  ): Promise<AutocompleteResult> => {
    const response = await apiClient.get<AutocompleteResult>(
      `/api/autocomplete/${field}`,
      {
        params: { q: query, limit, fuzzy },
        timeout: 5000, // Quick timeout for autocomplete
      }
    );
    return response.data;
  },

  // Semantic search for natural language queries
  semanticSearch: async (
    query: string,
    options: {
      limit?: number;
      state?: string;
      city?: string;
      minMoves?: number;
      customerType?: 'BUSINESS' | 'INDIVIDUAL';
    } = {}
  ): Promise<SemanticSearchResult> => {
    const { limit = 10, state, city, minMoves, customerType } = options;
    const response = await apiClient.get<SemanticSearchResult>(
      '/api/search/semantic',
      {
        params: {
          q: query,
          limit,
          state,
          city,
          min_moves: minMoves,
          customer_type: customerType,
        },
      }
    );
    return response.data;
  },

  // Single address verification endpoint
  verifySingleAddress: async (
    address: string,
    context?: Record<string, string>
  ): Promise<SingleAddressVerificationResponse> => {
    const response = await apiClient.post<SingleAddressVerificationResponse>(
      '/api/address/verify',
      { address, context: context || {} }
    );
    return response.data;
  },

  // Street name suggestions for address completion
  suggestStreet: async (
    partial: string,
    zipCode: string,
    streetNumber?: string,
    limit: number = 5
  ): Promise<{ suggestions: StreetSuggestion[]; zip_code: string; partial: string }> => {
    const response = await apiClient.post<{
      suggestions: StreetSuggestion[];
      zip_code: string;
      partial: string;
    }>('/api/address/suggest/street', {
      partial,
      zip_code: zipCode,
      street_number: streetNumber,
      limit,
    });
    return response.data;
  },

  // Knowledge Base API
  knowledgeUpload: async (
    file: File,
    tags: string,
    category: string
  ): Promise<KnowledgeUploadResult> => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('tags', tags);
    formData.append('category', category);

    const response = await apiClient.post<KnowledgeUploadResult>(
      '/api/knowledge/upload',
      formData,
      {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 120000,
      }
    );
    return response.data;
  },

  knowledgeListDocuments: async (
    category?: string
  ): Promise<{ documents: KnowledgeDocument[] }> => {
    const response = await apiClient.get<{ documents: KnowledgeDocument[] }>(
      '/api/knowledge/documents',
      { params: category ? { category } : {} }
    );
    return response.data;
  },

  knowledgeStats: async (): Promise<KnowledgeStats> => {
    const response = await apiClient.get<KnowledgeStats>('/api/knowledge/stats');
    return response.data;
  },

  knowledgeDelete: async (docId: string): Promise<{ deleted: number }> => {
    const response = await apiClient.delete<{ deleted: number }>(
      `/api/knowledge/documents/${docId}`
    );
    return response.data;
  },

  knowledgeSearch: async (
    query: string,
    options: { limit?: number; tags?: string[]; category?: string } = {}
  ): Promise<KnowledgeSearchResult> => {
    const response = await apiClient.post<KnowledgeSearchResult>(
      '/api/knowledge/search',
      { query, ...options }
    );
    return response.data;
  },

  // Download results by token (for paginated large datasets)
  downloadResults: async (
    token: string,
    format: 'json' | 'csv' = 'json'
  ): Promise<DownloadResult | Blob> => {
    if (format === 'csv') {
      const response = await apiClient.get(`/api/query/download/${token}`, {
        params: { format: 'csv' },
        responseType: 'blob',
        timeout: 60000,
      });
      return response.data;
    }
    const response = await apiClient.get<DownloadResult>(
      `/api/query/download/${token}`,
      {
        params: { format: 'json' },
        timeout: 60000,
      }
    );
    return response.data;
  },

  // Paginate results by token (for inline "See More" functionality)
  paginateResults: async (
    token: string,
    offset: number = 0,
    limit: number = 15
  ): Promise<{
    success: boolean;
    data: Record<string, unknown>[];
    offset: number;
    limit: number;
    total: number;
    has_more: boolean;
    remaining: number;
  }> => {
    const response = await apiClient.get(`/api/query/paginate/${token}`, {
      params: { offset, limit },
      timeout: 30000,
    });
    return response.data;
  },

  // Helper to trigger file download in browser
  triggerDownload: (data: Blob | DownloadResult, filename: string, format: 'json' | 'csv') => {
    let blob: Blob;
    if (data instanceof Blob) {
      blob = data;
    } else {
      const content = format === 'csv'
        ? convertToCSV(data.data)
        : JSON.stringify(data, null, 2);
      blob = new Blob([content], {
        type: format === 'csv' ? 'text/csv' : 'application/json',
      });
    }
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${filename}.${format}`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  },

  // ============================================================================
  // Enhanced Address Validation API
  // ============================================================================

  // Enhanced address validation with detailed scoring
  validateAddressEnhanced: async (
    address: string,
    mode: 'validate' | 'complete' | 'correct' | 'standardize' = 'correct',
    context?: Record<string, string>
  ): Promise<EnhancedValidationResponse> => {
    const response = await apiClient.post<EnhancedValidationResponse>(
      '/api/address/validate',
      { address, mode, context: context || {} }
    );
    return response.data;
  },

  // Quick validation check
  validateAddressQuick: async (
    address: string,
    mode: 'validate' | 'complete' | 'correct' | 'standardize' = 'correct'
  ): Promise<QuickValidationResponse> => {
    const response = await apiClient.get<QuickValidationResponse>(
      '/api/address/validate/quick',
      { params: { address, mode } }
    );
    return response.data;
  },

  // Complete a partial address
  completeAddress: async (
    address: string,
    context?: Record<string, string>
  ): Promise<{
    success: boolean;
    original: Record<string, unknown>;
    completed: Record<string, unknown> | null;
    standardized: string | null;
    completions_made: string[];
    confidence: number;
    confidence_percent: number;
    is_complete: boolean;
  }> => {
    const response = await apiClient.post('/api/address/complete', null, {
      params: { address, ...(context || {}) },
    });
    return response.data;
  },

  // Correct errors in an address
  correctAddress: async (
    address: string,
    context?: Record<string, string>
  ): Promise<{
    success: boolean;
    original: Record<string, unknown>;
    corrected: Record<string, unknown> | null;
    standardized: string | null;
    corrections_made: string[];
    confidence: number;
    confidence_percent: number;
    was_corrected: boolean;
  }> => {
    const response = await apiClient.post('/api/address/correct', null, {
      params: { address, ...(context || {}) },
    });
    return response.data;
  },

  // Standardize address to USPS format
  standardizeAddress: async (
    address: string
  ): Promise<{
    success: boolean;
    original: string;
    standardized: string | null;
    confidence: number;
    is_puerto_rico: boolean;
  }> => {
    const response = await apiClient.post('/api/address/standardize', null, {
      params: { address },
    });
    return response.data;
  },

  // Batch validation with enhanced scoring
  validateAddressBatch: async (
    addresses: string[],
    mode: 'validate' | 'complete' | 'correct' | 'standardize' = 'correct'
  ): Promise<{
    success: boolean;
    total: number;
    valid_count: number;
    deliverable_count: number;
    valid_rate: number;
    deliverable_rate: number;
    average_confidence: number;
    average_confidence_percent: number;
    results: Array<{
      address: string;
      is_valid: boolean;
      is_deliverable: boolean;
      confidence: number;
      confidence_percent: number;
      status: string;
      standardized: string | null;
      corrections_count: number;
      issues_count: number;
    }>;
  }> => {
    const response = await apiClient.post('/api/address/validate/batch', addresses, {
      params: { mode },
    });
    return response.data;
  },
};

// Additional type for street suggestions
interface StreetSuggestion {
  street_name: string;
  street_type: string;
  city: string;
}

export default api;
