import React, { useState } from 'react';
import {
  Dialog, DialogTitle, DialogContent, DialogActions,
  Button, TextField, Box, Typography, IconButton,
  CircularProgress, Chip, Collapse, List, ListItem, ListItemText,
  LinearProgress, Tabs, Tab, Tooltip, Paper, Divider,
  ToggleButton, ToggleButtonGroup, Alert,
} from '@mui/material';
import {
  Close as CloseIcon,
  VerifiedUser as ValidatorIcon,
  ExpandMore as ExpandIcon,
  ExpandLess as CollapseIcon,
  CheckCircle as CheckIcon,
  Warning as WarningIcon,
  Error as ErrorIcon,
  Info as InfoIcon,
  AutoFixHigh as CorrectIcon,
  Lightbulb as SuggestIcon,
  ContentCopy as CopyIcon,
} from '@mui/icons-material';
import { alpha } from '@mui/material/styles';
import { colors, borderRadius } from '../theme';
import { api } from '../services/api';
import type { EnhancedValidationResponse, ComponentScore, ValidationIssue } from '../types';

interface ValidatorProps {
  open: boolean;
  onClose: () => void;
}

type ValidationMode = 'validate' | 'complete' | 'correct' | 'standardize';

// Component confidence color mapping
const getConfidenceColor = (confidence: string): string => {
  switch (confidence) {
    case 'exact': return colors.success.main;
    case 'high': return colors.success.light;
    case 'medium': return colors.warning.main;
    case 'low': return colors.error.light;
    case 'inferred': return colors.info.main;
    case 'missing': return colors.text.disabled;
    default: return colors.text.secondary;
  }
};

// Status color mapping
const getStatusColor = (status: string): string => {
  switch (status) {
    case 'verified': return colors.success.main;
    case 'corrected':
    case 'completed': return colors.warning.main;
    case 'suggested': return colors.info.main;
    case 'unverified':
    case 'failed': return colors.error.main;
    default: return colors.text.secondary;
  }
};

// Quality badge color
const getQualityColor = (quality: string): string => {
  switch (quality) {
    case 'complete': return colors.success.main;
    case 'partial': return colors.warning.main;
    case 'ambiguous': return colors.info.main;
    case 'invalid': return colors.error.main;
    default: return colors.text.secondary;
  }
};

// Severity icon
const SeverityIcon: React.FC<{ severity: string }> = ({ severity }) => {
  switch (severity) {
    case 'error': return <ErrorIcon sx={{ color: colors.error.main, fontSize: 18 }} />;
    case 'warning': return <WarningIcon sx={{ color: colors.warning.main, fontSize: 18 }} />;
    case 'info': return <InfoIcon sx={{ color: colors.info.main, fontSize: 18 }} />;
    default: return null;
  }
};

// Confidence bar component
const ConfidenceBar: React.FC<{ value: number; label?: string }> = ({ value, label }) => {
  const getBarColor = (v: number): string => {
    if (v >= 0.85) return colors.success.main;
    if (v >= 0.70) return colors.warning.main;
    return colors.error.main;
  };

  return (
    <Box sx={{ width: '100%' }}>
      {label && (
        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
          <Typography variant="caption" color="text.secondary">{label}</Typography>
          <Typography variant="caption" fontWeight="bold" sx={{ color: getBarColor(value) }}>
            {Math.round(value * 100)}%
          </Typography>
        </Box>
      )}
      <LinearProgress
        variant="determinate"
        value={value * 100}
        sx={{
          height: 8,
          borderRadius: 4,
          backgroundColor: alpha(colors.text.disabled, 0.2),
          '& .MuiLinearProgress-bar': {
            backgroundColor: getBarColor(value),
            borderRadius: 4,
          },
        }}
      />
    </Box>
  );
};

// Component score display
const ComponentScoreDisplay: React.FC<{ score: ComponentScore }> = ({ score }) => {
  const formatComponent = (c: string): string => {
    return c.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
  };

  return (
    <Box
      sx={{
        display: 'flex',
        alignItems: 'center',
        gap: 1,
        py: 0.5,
        px: 1,
        borderRadius: 1,
        backgroundColor: alpha(getConfidenceColor(score.confidence), 0.1),
        border: `1px solid ${alpha(getConfidenceColor(score.confidence), 0.3)}`,
      }}
    >
      <Box sx={{ flex: 1, minWidth: 0 }}>
        <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
          {formatComponent(score.component)}
        </Typography>
        <Typography variant="body2" noWrap>
          {score.validated_value || score.original_value || '-'}
        </Typography>
      </Box>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
        {score.was_corrected && (
          <Tooltip title={score.correction_reason || 'Corrected'}>
            <CorrectIcon sx={{ color: colors.warning.main, fontSize: 16 }} />
          </Tooltip>
        )}
        {score.was_completed && (
          <Tooltip title="Completed">
            <SuggestIcon sx={{ color: colors.info.main, fontSize: 16 }} />
          </Tooltip>
        )}
        <Chip
          label={`${Math.round(score.score * 100)}%`}
          size="small"
          sx={{
            backgroundColor: getConfidenceColor(score.confidence),
            color: 'white',
            fontWeight: 'bold',
            fontSize: '0.7rem',
            height: 20,
          }}
        />
      </Box>
    </Box>
  );
};

// Build address string from component scores when standardized is not available
const buildAddressFromComponents = (scores: ComponentScore[]): string => {
  const getValue = (component: string): string => {
    const score = scores.find(s => s.component === component);
    return score?.validated_value || score?.original_value || '';
  };

  const parts: string[] = [];

  // Urbanization (PR only)
  const urb = getValue('urbanization');
  if (urb) parts.push(`URB ${urb}`);

  // Street line
  const streetParts = [
    getValue('street_number'),
    getValue('street_name'),
    getValue('street_type'),
  ].filter(Boolean);
  if (streetParts.length > 0) parts.push(streetParts.join(' '));

  // Unit
  const unit = getValue('unit');
  if (unit) parts.push(unit);

  // City, State ZIP
  const city = getValue('city');
  const state = getValue('state');
  const zip = getValue('zip_code');

  const cszParts: string[] = [];
  if (city) cszParts.push(city);
  if (state) cszParts.push(state);
  if (zip) cszParts.push(zip);

  if (cszParts.length > 0) {
    if (city && (state || zip)) {
      parts.push(`${city}, ${[state, zip].filter(Boolean).join(' ')}`);
    } else {
      parts.push(cszParts.join(' '));
    }
  }

  return parts.join(', ') || 'Address components shown below';
};

export const Validator: React.FC<ValidatorProps> = ({ open, onClose }) => {
  const [address, setAddress] = useState('');
  const [mode, setMode] = useState<ValidationMode>('correct');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<EnhancedValidationResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState(0);
  const [showAlternatives, setShowAlternatives] = useState(false);
  const [showCorrections, setShowCorrections] = useState(false);
  const [copied, setCopied] = useState(false);

  const handleValidate = async () => {
    if (!address.trim()) return;
    setLoading(true);
    setResult(null);
    setError(null);
    setActiveTab(0);

    try {
      const response = await api.validateAddressEnhanced(address, mode);
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
    setShowCorrections(false);
    setActiveTab(0);
    onClose();
  };

  const handleCopy = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleModeChange = (_: React.MouseEvent<HTMLElement>, newMode: ValidationMode | null) => {
    if (newMode) setMode(newMode);
  };

  // Group issues by severity
  const groupedIssues = result?.issues.reduce((acc, issue) => {
    if (!acc[issue.severity]) acc[issue.severity] = [];
    acc[issue.severity].push(issue);
    return acc;
  }, {} as Record<string, ValidationIssue[]>) || {};

  return (
    <Dialog
      open={open}
      onClose={handleClose}
      maxWidth="md"
      fullWidth
      PaperProps={{
        sx: {
          backgroundColor: colors.background.paper,
          borderRadius: borderRadius.lg,
          maxHeight: '90vh',
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
        {/* Mode Selection */}
        <Box sx={{ mb: 2 }}>
          <Typography variant="caption" color="text.secondary" sx={{ mb: 1, display: 'block' }}>
            Validation Mode
          </Typography>
          <ToggleButtonGroup
            value={mode}
            exclusive
            onChange={handleModeChange}
            size="small"
            fullWidth
          >
            <ToggleButton value="validate">
              <Tooltip title="Check validity only, no corrections">
                <span>Validate</span>
              </Tooltip>
            </ToggleButton>
            <ToggleButton value="complete">
              <Tooltip title="Fill in missing components">
                <span>Complete</span>
              </Tooltip>
            </ToggleButton>
            <ToggleButton value="correct">
              <Tooltip title="Fix errors and complete (recommended)">
                <span>Correct</span>
              </Tooltip>
            </ToggleButton>
            <ToggleButton value="standardize">
              <Tooltip title="Format to USPS standard">
                <span>Standardize</span>
              </Tooltip>
            </ToggleButton>
          </ToggleButtonGroup>
        </Box>

        {/* Address Input */}
        <TextField
          fullWidth
          multiline
          rows={2}
          placeholder="Enter address to validate (e.g., '101 turkey 22222' or '123 Main Stret, New York, NY')"
          value={address}
          onChange={(e) => setAddress(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              handleValidate();
            }
          }}
          sx={{ mt: 1 }}
        />

        {/* Error Display */}
        {error && (
          <Alert severity="error" sx={{ mt: 2 }}>
            {error}
          </Alert>
        )}

        {/* Result Display */}
        {result && (
          <Box sx={{ mt: 2 }}>
            {/* Status Header */}
            <Paper
              elevation={0}
              sx={{
                p: 2,
                borderRadius: borderRadius.md,
                backgroundColor: alpha(getStatusColor(result.status), 0.1),
                border: `1px solid ${getStatusColor(result.status)}`,
              }}
            >
              {/* Status Row */}
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2, flexWrap: 'wrap' }}>
                {result.is_valid ? (
                  <CheckIcon sx={{ color: colors.success.main }} />
                ) : (
                  <ErrorIcon sx={{ color: colors.error.main }} />
                )}
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
                  label={result.quality.toUpperCase()}
                  size="small"
                  variant="outlined"
                  sx={{ borderColor: getQualityColor(result.quality), color: getQualityColor(result.quality) }}
                />
                {result.is_deliverable && (
                  <Chip label="DELIVERABLE" size="small" color="success" variant="outlined" />
                )}
                {result.is_puerto_rico && (
                  <Chip label="PUERTO RICO" size="small" color="info" variant="outlined" />
                )}
              </Box>

              {/* Confidence Bar */}
              <ConfidenceBar value={result.overall_confidence} label="Overall Confidence" />

              {/* Standardized/Inferred Address Display */}
              {(result.standardized || result.component_scores.length > 0) && (
                <Box sx={{ mt: 2 }}>
                  <Typography variant="caption" color="text.secondary">
                    {result.status === 'verified' ? 'Verified Address:' :
                     result.status === 'corrected' ? 'Corrected Address:' :
                     result.status === 'completed' ? 'Completed Address:' :
                     result.status === 'suggested' ? 'Suggested Address:' :
                     'Inferred Address:'}
                  </Typography>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 0.5 }}>
                    <Typography
                      variant="body1"
                      sx={{
                        fontWeight: 'medium',
                        fontFamily: 'monospace',
                        backgroundColor: alpha(colors.background.default, 0.5),
                        p: 1,
                        borderRadius: 1,
                        flex: 1,
                      }}
                    >
                      {result.standardized || buildAddressFromComponents(result.component_scores)}
                    </Typography>
                    <Tooltip title={copied ? 'Copied!' : 'Copy to clipboard'}>
                      <IconButton size="small" onClick={() => handleCopy(result.standardized || buildAddressFromComponents(result.component_scores))}>
                        <CopyIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                  </Box>
                </Box>
              )}

              {/* Puerto Rico Urbanization Status */}
              {result.is_puerto_rico && result.urbanization_status && (
                <Box sx={{ mt: 1 }}>
                  <Typography variant="caption" color="text.secondary">
                    Urbanization: {' '}
                    <Chip
                      label={result.urbanization_status.toUpperCase()}
                      size="small"
                      sx={{
                        backgroundColor: result.urbanization_status === 'present'
                          ? colors.success.main
                          : result.urbanization_status === 'inferred'
                          ? colors.info.main
                          : colors.warning.main,
                        color: 'white',
                        height: 18,
                        fontSize: '0.65rem',
                      }}
                    />
                  </Typography>
                  {result.urbanization_status === 'missing' && (
                    <Typography variant="caption" sx={{ color: colors.warning.main, display: 'block', mt: 0.5 }}>
                      Puerto Rico addresses require urbanization (URB) for reliable delivery
                    </Typography>
                  )}
                </Box>
              )}
            </Paper>

            {/* Tabs for Details */}
            <Box sx={{ mt: 2 }}>
              <Tabs value={activeTab} onChange={(_, v) => setActiveTab(v)} variant="fullWidth">
                <Tab label="Components" />
                <Tab label={`Issues (${result.issues.length})`} />
                <Tab label="Changes" />
              </Tabs>

              {/* Components Tab */}
              {activeTab === 0 && (
                <Box sx={{ mt: 2 }}>
                  <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: 1 }}>
                    {result.component_scores.map((score, idx) => (
                      <ComponentScoreDisplay key={idx} score={score} />
                    ))}
                  </Box>
                </Box>
              )}

              {/* Issues Tab */}
              {activeTab === 1 && (
                <Box sx={{ mt: 2 }}>
                  {result.issues.length === 0 ? (
                    <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center', py: 2 }}>
                      No issues found
                    </Typography>
                  ) : (
                    <List dense>
                      {['error', 'warning', 'info'].map((severity) => (
                        groupedIssues[severity]?.map((issue, idx) => (
                          <ListItem
                            key={`${severity}-${idx}`}
                            sx={{
                              backgroundColor: alpha(
                                severity === 'error' ? colors.error.main :
                                severity === 'warning' ? colors.warning.main :
                                colors.info.main,
                                0.1
                              ),
                              borderRadius: 1,
                              mb: 0.5,
                            }}
                          >
                            <SeverityIcon severity={severity} />
                            <ListItemText
                              primary={issue.message}
                              secondary={issue.suggestion}
                              sx={{ ml: 1 }}
                              primaryTypographyProps={{ variant: 'body2' }}
                              secondaryTypographyProps={{ variant: 'caption' }}
                            />
                            {issue.auto_fixable && (
                              <Chip label="Auto-fixable" size="small" color="info" variant="outlined" sx={{ ml: 1 }} />
                            )}
                          </ListItem>
                        ))
                      ))}
                    </List>
                  )}
                </Box>
              )}

              {/* Changes Tab */}
              {activeTab === 2 && (
                <Box sx={{ mt: 2 }}>
                  {/* Corrections */}
                  {result.corrections_applied.length > 0 && (
                    <Box sx={{ mb: 2 }}>
                      <Button
                        size="small"
                        onClick={() => setShowCorrections(!showCorrections)}
                        startIcon={<CorrectIcon />}
                        endIcon={showCorrections ? <CollapseIcon /> : <ExpandIcon />}
                      >
                        {result.corrections_applied.length} Correction{result.corrections_applied.length > 1 ? 's' : ''} Applied
                      </Button>
                      <Collapse in={showCorrections}>
                        <List dense sx={{ mt: 1 }}>
                          {result.corrections_applied.map((correction, idx) => (
                            <ListItem key={idx} sx={{ py: 0 }}>
                              <ListItemText
                                primary={correction}
                                primaryTypographyProps={{ variant: 'body2' }}
                              />
                            </ListItem>
                          ))}
                        </List>
                      </Collapse>
                    </Box>
                  )}

                  {/* Completions */}
                  {result.completions_applied.length > 0 && (
                    <Box>
                      <Typography variant="subtitle2" sx={{ mb: 1 }}>
                        <SuggestIcon sx={{ fontSize: 16, mr: 0.5, verticalAlign: 'middle' }} />
                        Completions Applied
                      </Typography>
                      <List dense>
                        {result.completions_applied.map((completion, idx) => (
                          <ListItem key={idx} sx={{ py: 0 }}>
                            <ListItemText
                              primary={completion}
                              primaryTypographyProps={{ variant: 'body2' }}
                            />
                          </ListItem>
                        ))}
                      </List>
                    </Box>
                  )}

                  {result.corrections_applied.length === 0 && result.completions_applied.length === 0 && (
                    <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center', py: 2 }}>
                      No changes applied
                    </Typography>
                  )}
                </Box>
              )}
            </Box>

            {/* Alternatives */}
            {result.alternatives.length > 0 && (
              <Box sx={{ mt: 2 }}>
                <Divider sx={{ my: 1 }} />
                <Button
                  size="small"
                  onClick={() => setShowAlternatives(!showAlternatives)}
                  endIcon={showAlternatives ? <CollapseIcon /> : <ExpandIcon />}
                >
                  {result.alternatives.length} Alternative{result.alternatives.length > 1 ? 's' : ''} Found
                </Button>
                <Collapse in={showAlternatives}>
                  <List dense sx={{ mt: 1 }}>
                    {result.alternatives.map((alt, idx) => (
                      <ListItem
                        key={idx}
                        sx={{ py: 0.5 }}
                        secondaryAction={
                          <IconButton
                            size="small"
                            onClick={() => {
                              const altLine = alt.single_line as string ||
                                `${alt.street_number || ''} ${alt.street_name || ''} ${alt.street_type || ''}, ${alt.city || ''}, ${alt.state || ''} ${alt.zip_code || ''}`.trim();
                              setAddress(altLine);
                            }}
                          >
                            <CopyIcon fontSize="small" />
                          </IconButton>
                        }
                      >
                        <ListItemText
                          primary={
                            alt.single_line as string ||
                            `${alt.street_number || ''} ${alt.street_name || ''} ${alt.street_type || ''}, ${alt.city || ''}, ${alt.state || ''} ${alt.zip_code || ''}`.trim()
                          }
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
