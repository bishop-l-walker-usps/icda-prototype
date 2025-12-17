import React, { useState } from 'react';
import {
  Dialog, DialogTitle, DialogContent, DialogActions,
  Button, TextField, Box, Typography, IconButton,
  CircularProgress, Chip, Collapse, List, ListItem, ListItemText,
} from '@mui/material';
import {
  Close as CloseIcon,
  VerifiedUser as ValidatorIcon,
  ExpandMore as ExpandIcon,
  ExpandLess as CollapseIcon,
} from '@mui/icons-material';
import { alpha } from '@mui/material/styles';
import { colors, borderRadius } from '../theme';
import { api } from '../services/api';
import type { SingleAddressVerificationResponse } from '../types';

interface ValidatorProps {
  open: boolean;
  onClose: () => void;
}

export const Validator: React.FC<ValidatorProps> = ({ open, onClose }) => {
  const [address, setAddress] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<SingleAddressVerificationResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showAlternatives, setShowAlternatives] = useState(false);

  const handleValidate = async () => {
    if (!address.trim()) return;
    setLoading(true);
    setResult(null);
    setError(null);

    try {
      const response = await api.verifySingleAddress(address);
      setResult(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Validation failed - check if backend is running');
    } finally {
      setLoading(false);
    }
  };

  const handleClose = () => {
    setAddress('');
    setResult(null);
    setError(null);
    setShowAlternatives(false);
    onClose();
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'verified':
        return colors.success.main;
      case 'corrected':
      case 'completed':
        return colors.warning.main;
      case 'unverified':
      case 'failed':
        return colors.error.main;
      default:
        return colors.text.secondary;
    }
  };

  return (
    <Dialog
      open={open}
      onClose={handleClose}
      maxWidth="sm"
      fullWidth
      PaperProps={{
        sx: {
          backgroundColor: colors.background.paper,
          borderRadius: borderRadius.lg,
        },
      }}
    >
      <DialogTitle sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <ValidatorIcon sx={{ color: colors.success.main }} />
          <Typography variant="h6">Address Validator</Typography>
        </Box>
        <IconButton onClick={handleClose} size="small" aria-label="Close dialog">
          <CloseIcon />
        </IconButton>
      </DialogTitle>

      <DialogContent>
        <TextField
          fullWidth
          multiline
          rows={3}
          placeholder="Enter address to validate..."
          value={address}
          onChange={(e) => setAddress(e.target.value)}
          sx={{ mt: 1 }}
        />

        {/* Error Display */}
        {error && (
          <Box
            sx={{
              mt: 2,
              p: 2,
              borderRadius: borderRadius.md,
              backgroundColor: alpha(colors.error.main, 0.1),
              border: `1px solid ${colors.error.main}`,
            }}
          >
            <Typography sx={{ color: colors.error.light }}>
              {error}
            </Typography>
          </Box>
        )}

        {/* Result Display */}
        {result && (
          <Box sx={{ mt: 2 }}>
            {/* Status Header */}
            <Box
              sx={{
                p: 2,
                borderRadius: borderRadius.md,
                backgroundColor: alpha(getStatusColor(result.status), 0.1),
                border: `1px solid ${getStatusColor(result.status)}`,
              }}
            >
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                <Chip
                  label={result.status.toUpperCase()}
                  size="small"
                  sx={{
                    backgroundColor: getStatusColor(result.status),
                    color: 'white',
                    fontWeight: 'bold',
                  }}
                />
                <Chip
                  label={`${Math.round(result.confidence * 100)}% confidence`}
                  size="small"
                  variant="outlined"
                />
                {result.match_type && (
                  <Chip label={result.match_type} size="small" variant="outlined" />
                )}
              </Box>

              {/* Verified Address */}
              {result.verified && (
                <Box sx={{ mt: 1 }}>
                  <Typography variant="caption" color="text.secondary">
                    {result.status === 'verified' ? 'Verified Address:' :
                     result.status === 'corrected' ? 'Corrected Address:' :
                     result.status === 'suggested' ? 'Suggested Address (review required):' :
                     'Best Match:'}
                  </Typography>
                  <Typography variant="body2" sx={{ fontWeight: 'medium' }}>
                    {String(result.verified.single_line ||
                      `${result.verified.street_number || ''} ${result.verified.street_name || ''} ${result.verified.street_type || ''}, ${result.verified.city || ''}, ${result.verified.state || ''} ${result.verified.zip_code || ''}`.trim())}
                  </Typography>
                </Box>
              )}

              {/* Unverified - show reason */}
              {(result.status === 'unverified' || result.status === 'failed') && (() => {
                const reason = result.metadata?.reason as string | undefined;
                const suggestion = result.metadata?.suggestion as string | undefined;
                const bestMatch = result.metadata?.best_match as string | undefined;
                return (
                  <Box sx={{ mt: 1 }}>
                    <Typography variant="body2" sx={{ color: colors.error.light }}>
                      {reason || 'This address could not be verified against our database.'}
                    </Typography>
                    {suggestion && (
                      <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.5 }}>
                        Tip: {suggestion}
                      </Typography>
                    )}
                    {bestMatch && (
                      <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.5 }}>
                        Closest match found: {bestMatch}
                      </Typography>
                    )}
                  </Box>
                );
              })()}

              {/* Puerto Rico Warnings */}
              {result.pr_warnings.length > 0 && (
                <Box sx={{ mt: 1 }}>
                  {result.pr_warnings.map((warning, idx) => (
                    <Typography
                      key={idx}
                      variant="caption"
                      sx={{ color: colors.warning.main, display: 'block' }}
                    >
                      ⚠️ {warning}
                    </Typography>
                  ))}
                </Box>
              )}

              {/* Processing Time */}
              <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                Processed in {result.processing_time_ms}ms
              </Typography>
            </Box>

            {/* Alternatives */}
            {result.alternatives.length > 0 && (
              <Box sx={{ mt: 2 }}>
                <Button
                  size="small"
                  onClick={() => setShowAlternatives(!showAlternatives)}
                  endIcon={showAlternatives ? <CollapseIcon /> : <ExpandIcon />}
                >
                  {result.alternatives.length} Alternative{result.alternatives.length > 1 ? 's' : ''}
                </Button>
                <Collapse in={showAlternatives}>
                  <List dense sx={{ mt: 1 }}>
                    {result.alternatives.map((alt, idx) => (
                      <ListItem key={idx} sx={{ py: 0 }}>
                        <ListItemText
                          primary={String(alt.single_line || 
                            `${alt.street_number || ''} ${alt.street_name || ''} ${alt.street_type || ''}, ${alt.city || ''}, ${alt.state || ''} ${alt.zip_code || ''}`.trim())}
                          primaryTypographyProps={{ variant: 'body2' }}
                        />
                      </ListItem>
                    ))}
                  </List>
                </Collapse>
              </Box>
            )}
          </Box>
        )}
      </DialogContent>

      <DialogActions sx={{ p: 2 }}>
        <Button onClick={handleClose} color="inherit">
          Cancel
        </Button>
        <Button
          variant="contained"
          onClick={handleValidate}
          disabled={loading || !address.trim()}
          startIcon={loading ? <CircularProgress size={16} /> : <ValidatorIcon />}
          sx={{
            backgroundColor: colors.success.main,
            '&:hover': { backgroundColor: colors.success.dark },
          }}
        >
          Validate
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default Validator;
