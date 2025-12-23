import React, { useState } from 'react';
import { Box, Typography, Card, CardContent, Chip, Stack, IconButton, Tooltip, CircularProgress, Snackbar, Alert } from '@mui/material';
import {
  Cloud as CloudIcon, Storage as StorageIcon, Memory as MemoryIcon, Search as SearchIcon,
  CheckCircle as CheckIcon, Cancel as CancelIcon, HourglassEmpty as LoadingIcon,
  LibraryBooks as KnowledgeIcon, Refresh as RefreshIcon,
} from '@mui/icons-material';
import { colors } from '../theme';
import { getStatusColor, getStatusBgColor, type Status } from '../utils';
import type { HealthStatus } from '../types';
import { KnowledgeUploadModal } from './KnowledgeUploadModal';

interface AWSToolingPanelProps {
  health: HealthStatus | null;
  loading: boolean;
}

const SERVICES = [
  { name: 'Bedrock Nova', desc: 'NLP Query Processing', icon: <CloudIcon />, key: 'nova' as const },
  { name: 'Titan Embeddings', desc: 'Vector Embeddings', icon: <MemoryIcon />, key: 'embedder' as const },
  { name: 'ElastiCache', desc: 'Redis Caching', icon: <StorageIcon />, key: 'redis' as const },
  { name: 'OpenSearch', desc: 'Vector Index', icon: <SearchIcon />, key: 'opensearch' as const },
] as const;

const STATUS_ICONS = { online: CheckIcon, offline: CancelIcon, loading: LoadingIcon };

const StatusIcon: React.FC<{ status: Status }> = ({ status }) => {
  const Icon = STATUS_ICONS[status];
  return <Icon sx={{ color: getStatusColor(status), fontSize: 16 }} />;
};

export const AWSToolingPanel: React.FC<AWSToolingPanelProps> = ({ health, loading }) => {
  const [knowledgeModalOpen, setKnowledgeModalOpen] = useState(false);
  const [reindexing, setReindexing] = useState(false);
  const [snackbar, setSnackbar] = useState<{ open: boolean; message: string; severity: 'success' | 'error' }>({
    open: false,
    message: '',
    severity: 'success',
  });

  const handleReindex = async () => {
    setReindexing(true);
    try {
      const response = await fetch('/api/knowledge/reindex', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      });
      const result = await response.json();

      if (result.success) {
        const indexed = result.indexed || 0;
        const skipped = result.skipped || 0;
        setSnackbar({
          open: true,
          message: `Reindex complete: ${indexed} indexed, ${skipped} unchanged`,
          severity: 'success',
        });
      } else {
        setSnackbar({
          open: true,
          message: result.error || 'Reindex failed',
          severity: 'error',
        });
      }
    } catch (error) {
      setSnackbar({
        open: true,
        message: 'Failed to connect to server',
        severity: 'error',
      });
    } finally {
      setReindexing(false);
    }
  };

  return (
    <Box sx={{ p: 2 }}>
      <Typography variant="h6" sx={{ mb: 2, color: colors.aws.orange }}>AWS Tooling</Typography>
      <Stack spacing={1.5}>
        {SERVICES.map(({ name, desc, icon, key }) => {
          const status: Status = loading ? 'loading' : !health ? 'offline' : health[key] ? 'online' : 'offline';
          const isOpenSearch = key === 'opensearch';
          const isTitan = key === 'embedder';
          return (
            <Card key={name} sx={{ backgroundColor: 'background.elevated', border: 1, borderColor: getStatusBgColor(status, '44') }}>
              <CardContent sx={{ py: 1.5, px: 2, '&:last-child': { pb: 1.5 } }}>
                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
                    <Box sx={{ color: colors.aws.orange }}>{icon}</Box>
                    <Box>
                      <Typography variant="body2" fontWeight={600}>{name}</Typography>
                      <Typography variant="caption" color="text.secondary">{desc}</Typography>
                    </Box>
                  </Box>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                    {isTitan && (
                      <Tooltip title="Reindex Knowledge Base">
                        <span>
                          <IconButton
                            size="small"
                            onClick={handleReindex}
                            disabled={reindexing}
                            sx={{
                              color: colors.accent.main,
                              '&:hover': { backgroundColor: `${colors.accent.main}22` },
                              '&.Mui-disabled': { color: 'text.disabled' },
                            }}
                          >
                            {reindexing ? (
                              <CircularProgress size={16} color="inherit" />
                            ) : (
                              <RefreshIcon fontSize="small" />
                            )}
                          </IconButton>
                        </span>
                      </Tooltip>
                    )}
                    {isOpenSearch && (
                      <Tooltip title="Knowledge Base">
                        <IconButton
                          size="small"
                          onClick={() => setKnowledgeModalOpen(true)}
                          sx={{
                            color: colors.accent.main,
                            '&:hover': { backgroundColor: `${colors.accent.main}22` },
                          }}
                        >
                          <KnowledgeIcon fontSize="small" />
                        </IconButton>
                      </Tooltip>
                    )}
                    <Chip
                      icon={<StatusIcon status={status} />}
                      label={status}
                      size="small"
                      role="status"
                      aria-label={`${name} status: ${status}`}
                      sx={{
                        backgroundColor: getStatusBgColor(status),
                        color: getStatusColor(status),
                        textTransform: 'capitalize',
                        fontSize: '0.7rem',
                        height: 24,
                      }}
                    />
                  </Box>
                </Box>
              </CardContent>
            </Card>
          );
        })}
      </Stack>

      <KnowledgeUploadModal
        open={knowledgeModalOpen}
        onClose={() => setKnowledgeModalOpen(false)}
      />

      <Snackbar
        open={snackbar.open}
        autoHideDuration={4000}
        onClose={() => setSnackbar({ ...snackbar, open: false })}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert
          onClose={() => setSnackbar({ ...snackbar, open: false })}
          severity={snackbar.severity}
          variant="filled"
          sx={{ width: '100%' }}
        >
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default AWSToolingPanel;