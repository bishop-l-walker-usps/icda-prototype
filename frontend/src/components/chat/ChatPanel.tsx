import React, { useRef, useEffect } from 'react';
import { Box, Typography, CircularProgress } from '@mui/material';
import { styles } from '../../theme/styles';
import { MessageBubble } from './MessageBubble';
import type { ChatMessage } from '../../types';

interface ChatPanelProps {
  messages: ChatMessage[];
  loading: boolean;
}

export const ChatPanel: React.FC<ChatPanelProps> = ({ messages, loading }) => {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' });
  }, [messages]);

  return (
    <Box 
      sx={{ 
        display: 'flex', 
        flexDirection: 'column', 
        flex: 1,           // Take available space
        minHeight: 0,      // Critical for flex scroll
        overflow: 'hidden' // Contain the scroll area
      }}
    >
      <Box
        ref={scrollRef}
        sx={{ 
          ...styles.chat.container, 
          flex: 1,
          minHeight: 0,        // Critical for flex scroll
          overflowY: 'auto',   // Enable vertical scroll
          overflowX: 'hidden', // No horizontal scroll
        }}
      >
        {messages.length === 0 ? (
          <Box sx={{ ...styles.utils.flexCenter, flexGrow: 1, flexDirection: 'column', gap: 2, opacity: 0.5 }}>
            <Typography variant="h6">ICDA Query Interface</Typography>
            <Typography variant="body2" color="text.secondary">
              Ask questions about customer data. Try "Show me John Smith's account" or "List customers in Texas".
            </Typography>
          </Box>
        ) : (
          messages.map((msg) => <MessageBubble key={msg.id} message={msg} />)
        )}

        {loading && (
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, alignSelf: 'flex-start', p: 2 }}>
            <CircularProgress size={20} />
            <Typography variant="body2" color="text.secondary">Processing query...</Typography>
          </Box>
        )}
      </Box>
    </Box>
  );
};

export default ChatPanel;
