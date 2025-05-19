import React, { useEffect } from 'react';
const { AIPAgentClient } = require('@osdk/client');

const AuditChatbotWidget = () => {
    useEffect(() => {
        const agentClient = new AIPAgentClient({
            rid: 'ri.aip-agents..agent.4f4b0c55-2e80-4a97-abda-9abaca1144bd',
            auth: {
                clientId: '470f08785890cfedb4aa7fb356bf0278',
                redirectUri: 'http://localhost:8080/auth/callback',
                scope: 'openid profile aip:agent',
                authorizationUrl: 'https://auth.palantir.com/oauth2/authorize'
            }
        });

        async function initChatbot() {
            try {
                const token = await agentClient.auth.getAccessToken();
                const container = document.getElementById('aip-widget');
                if (token && container) {
                    container.innerHTML = `<div id="chatbot">Loading...</div>`;
                    console.log('Agent loaded with token:', token);
                } else {
                    container.innerHTML = 'Authentication failed—please log in.';
                }
            } catch (error) {
                console.error('OAuth2 error:', error);
                document.getElementById('aip-widget').innerHTML = 'Error loading chatbot.';
            }
        }

        initChatbot();
    }, []); // Empty dependency array to run once on mount

    return (
        <div className="chatbot-section">
            <h3>Ethical Audit Chatbot</h3>
            <div id="aip-widget"></div>
        </div>
    );
};

export default AuditChatbotWidget;