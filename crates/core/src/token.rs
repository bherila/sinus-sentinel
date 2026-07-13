//! Bearer-token storage behind a trait (SPEC §7 auth, §8 privacy). The desktop
//! app stores the PHR API key in the OS keychain ([`KeyringTokenStore`], behind
//! the `keyring` feature — Keychain on macOS, Credential Manager on Windows).
//! Tests use [`InMemoryTokenStore`]. Tokens never live in config files.

use std::sync::Mutex;

use crate::error::{Error, Result};

/// Read/write access to the PHR bearer token.
pub trait TokenStore: Send + Sync {
    fn get_token(&self) -> Result<Option<String>>;
    fn set_token(&self, token: &str) -> Result<()>;
    fn clear(&self) -> Result<()>;
}

/// In-memory token store for tests and ephemeral use.
#[derive(Debug, Default)]
pub struct InMemoryTokenStore {
    token: Mutex<Option<String>>,
}

impl InMemoryTokenStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_token(token: impl Into<String>) -> Self {
        InMemoryTokenStore {
            token: Mutex::new(Some(token.into())),
        }
    }
}

impl TokenStore for InMemoryTokenStore {
    fn get_token(&self) -> Result<Option<String>> {
        Ok(self
            .token
            .lock()
            .map_err(|_| Error::Token("poisoned".into()))?
            .clone())
    }

    fn set_token(&self, token: &str) -> Result<()> {
        *self
            .token
            .lock()
            .map_err(|_| Error::Token("poisoned".into()))? = Some(token.to_string());
        Ok(())
    }

    fn clear(&self) -> Result<()> {
        *self
            .token
            .lock()
            .map_err(|_| Error::Token("poisoned".into()))? = None;
        Ok(())
    }
}

/// OS keychain-backed token store (SPEC §7/§8).
#[cfg(feature = "keyring")]
pub struct KeyringTokenStore {
    service: String,
    account: String,
}

#[cfg(feature = "keyring")]
impl KeyringTokenStore {
    pub fn new(service: impl Into<String>, account: impl Into<String>) -> Self {
        KeyringTokenStore {
            service: service.into(),
            account: account.into(),
        }
    }

    fn entry(&self) -> Result<keyring::Entry> {
        keyring::Entry::new(&self.service, &self.account)
            .map_err(|e| Error::Token(format!("keyring: {e}")))
    }
}

#[cfg(feature = "keyring")]
impl TokenStore for KeyringTokenStore {
    fn get_token(&self) -> Result<Option<String>> {
        match self.entry()?.get_password() {
            Ok(t) => Ok(Some(t)),
            Err(keyring::Error::NoEntry) => Ok(None),
            Err(e) => Err(Error::Token(format!("keyring get: {e}"))),
        }
    }

    fn set_token(&self, token: &str) -> Result<()> {
        self.entry()?
            .set_password(token)
            .map_err(|e| Error::Token(format!("keyring set: {e}")))
    }

    fn clear(&self) -> Result<()> {
        match self.entry()?.delete_credential() {
            Ok(()) => Ok(()),
            Err(keyring::Error::NoEntry) => Ok(()),
            Err(e) => Err(Error::Token(format!("keyring clear: {e}"))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn in_memory_roundtrip() {
        let store = InMemoryTokenStore::new();
        assert!(store.get_token().unwrap().is_none());
        store.set_token("secret").unwrap();
        assert_eq!(store.get_token().unwrap().unwrap(), "secret");
        store.clear().unwrap();
        assert!(store.get_token().unwrap().is_none());
    }
}
