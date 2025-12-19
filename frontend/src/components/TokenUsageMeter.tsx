import React from 'react';
import {
  Box,
  Typography,
  LinearProgress,
  Tooltip,
  Collapse,
  IconButton,
} from '@mui/material';
import { alpha } from '@mui/material/styles';
import {
  ExpandMore as ExpandIcon,
  ExpandLess as CollapseIcon,
  Token as TokenIcon,
} from '@mui/icons-material';
import { colors, borderRadius } from '../theme';
import type { TokenUsage } from '../types';

interface TokenUsageMeterProps {
  usage: TokenUsage;
  compact?: boolean;
  showBreakdown?: boolean;
}

/**
 * Determines the color based on usage percentage.
 * - Green (<50%): Healthy usage
 * - Yellow (50-80%): Moderate usage
 * - Red (>80%): High usage, approaching limit
 */
const getUsageColor = (percentage: number): string => {
  if (percentage < 50) return colors.success.main;
  if (percentage < 80) return colors.warning.main;
  return colors.error.main;
};

/**
 * Formats a number with thousands separators.
 */
const formatNumber = (num: number): string => {
  return num.toLocaleString();
};

export const TokenUsageMeter: React.FC<TokenUsageMeterProps> = ({
  usage,
  compact = false,
  showBreakdown: initialShowBreakdown = false,
}) => {
  const [expanded, setExpanded] = React.useState(initialShowBreakdown);
  const percentage = usage.percentage_used ||
    (usage.context_limit > 0 ? (usage.total_tokens / usage.context_limit) * 100 : 0);
  const usageColor = getUsageColor(percentage);

  if (compact) {
    return (
      <Tooltip
        title={
          <Box sx={{ p: 0.5 }}>
            <Typography variant="caption" sx={{ fontWeight: 600 }}>
              Token Usage
            </Typography>
            <Box sx={{ mt: 0.5 }}>
              <Typography variant="caption" display="block">
                Input: {formatNumber(usage.input_tokens)}
              </Typography>
              <Typography variant="caption" display="block">
                Output: {formatNumber(usage.output_tokens)}
              </Typography>
              <Typography variant="caption" display="block">
                Total: {formatNumber(usage.total_tokens)} / {formatNumber(usage.context_limit)}
              </Typography>
            </Box>
          </Box>
        }
        arrow
        placement="top"
      >
        <Box
          sx={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: 0.5,
            px: 1,
            py: 0.25,
            borderRadius: borderRadius.sm,
            backgroundColor: alpha(usageColor, 0.15),
            border: `1px solid ${alpha(usageColor, 0.3)}`,
            cursor: 'default',
          }}
        >
          <TokenIcon sx={{ fontSize: 12, color: usageColor }} />
          <Typography
            variant="caption"
            sx={{
              fontSize: '0.65rem',
              fontWeight: 600,
              color: usageColor,
            }}
          >
            {percentage.toFixed(1)}%
          </Typography>
        </Box>
      </Tooltip>
    );
  }

  return (
    <Box
      sx={{
        p: 1.5,
        borderRadius: borderRadius.md,
        backgroundColor: alpha(colors.background.elevated, 0.5),
        border: `1px solid ${alpha(colors.text.primary, 0.1)}`,
      }}
    >
      {/* Header with progress bar */}
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <TokenIcon sx={{ fontSize: 16, color: usageColor }} />
        <Typography
          variant="caption"
          sx={{ fontWeight: 600, color: colors.text.primary }}
        >
          Token Usage
        </Typography>
        <Box sx={{ flex: 1 }} />
        <Typography
          variant="caption"
          sx={{
            fontWeight: 700,
            color: usageColor,
            fontSize: '0.75rem',
          }}
        >
          {percentage.toFixed(1)}%
        </Typography>
        <IconButton
          size="small"
          onClick={() => setExpanded(!expanded)}
          sx={{
            p: 0.25,
            color: colors.text.secondary,
            '&:hover': { color: colors.text.primary },
          }}
        >
          {expanded ? (
            <CollapseIcon sx={{ fontSize: 16 }} />
          ) : (
            <ExpandIcon sx={{ fontSize: 16 }} />
          )}
        </IconButton>
      </Box>

      {/* Progress bar */}
      <Box sx={{ mt: 1 }}>
        <LinearProgress
          variant="determinate"
          value={Math.min(percentage, 100)}
          sx={{
            height: 6,
            borderRadius: borderRadius.sm,
            backgroundColor: alpha(colors.text.primary, 0.1),
            '& .MuiLinearProgress-bar': {
              backgroundColor: usageColor,
              borderRadius: borderRadius.sm,
              transition: 'transform 0.4s ease, background-color 0.3s ease',
            },
          }}
        />
      </Box>

      {/* Expanded breakdown */}
      <Collapse in={expanded}>
        <Box
          sx={{
            mt: 1.5,
            pt: 1,
            borderTop: `1px solid ${alpha(colors.text.primary, 0.08)}`,
          }}
        >
          <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
            <Box sx={{ display: 'flex', gap: 2 }}>
              <Box>
                <Typography
                  variant="caption"
                  sx={{ color: colors.text.muted, fontSize: '0.65rem' }}
                >
                  Input
                </Typography>
                <Typography
                  variant="body2"
                  sx={{ fontWeight: 600, color: colors.info.light, fontSize: '0.75rem' }}
                >
                  {formatNumber(usage.input_tokens)}
                </Typography>
              </Box>
              <Box>
                <Typography
                  variant="caption"
                  sx={{ color: colors.text.muted, fontSize: '0.65rem' }}
                >
                  Output
                </Typography>
                <Typography
                  variant="body2"
                  sx={{ fontWeight: 600, color: colors.accent.light, fontSize: '0.75rem' }}
                >
                  {formatNumber(usage.output_tokens)}
                </Typography>
              </Box>
            </Box>
            <Box sx={{ textAlign: 'right' }}>
              <Typography
                variant="caption"
                sx={{ color: colors.text.muted, fontSize: '0.65rem' }}
              >
                Total / Limit
              </Typography>
              <Typography
                variant="body2"
                sx={{ fontWeight: 600, color: usageColor, fontSize: '0.75rem' }}
              >
                {formatNumber(usage.total_tokens)} / {formatNumber(usage.context_limit)}
              </Typography>
            </Box>
          </Box>
        </Box>
      </Collapse>
    </Box>
  );
};

export default TokenUsageMeter;
