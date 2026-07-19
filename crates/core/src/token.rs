//! Bearer-token storage behind a trait (SPEC §7 auth, §8 privacy). The desktop
//! app stores the PHR API key in the OS keychain ([`KeyringTokenStore`], behind
//! the `keyring` feature — Keychain on macOS, Credential Manager on Windows).
//! Tests use [`InMemoryTokenStore`]. Tokens never live in config files.

use std::sync::Mutex;

#[cfg(feature = "keyring")]
use std::collections::HashMap;
#[cfg(feature = "keyring")]
use std::sync::OnceLock;

use crate::error::{Error, Result};

/// Read/write access to the PHR bearer token.
pub trait TokenStore: Send + Sync {
    fn get_token(&self) -> Result<Option<String>>;
    fn set_token(&self, token: &str) -> Result<()>;
    fn clear(&self) -> Result<()>;
}

/// Forward the trait through a boxed store, so callers that pick an implementation
/// at runtime (e.g. keychain vs. in-memory by build feature) can hold a
/// `Box<dyn TokenStore>` and still satisfy `SyncEngine<T: TokenStore>`.
impl TokenStore for Box<dyn TokenStore> {
    fn get_token(&self) -> Result<Option<String>> {
        (**self).get_token()
    }

    fn set_token(&self, token: &str) -> Result<()> {
        (**self).set_token(token)
    }

    fn clear(&self) -> Result<()> {
        (**self).clear()
    }
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
type KeyringCache = HashMap<(String, String), std::result::Result<Option<String>, String>>;

/// Process-wide cache shared by the UI and sync worker. Besides avoiding repeated
/// Keychain IPC, caching failures prevents a denied credential from producing a
/// new macOS permission dialog on every backoff retry.
#[cfg(feature = "keyring")]
static KEYRING_CACHE: OnceLock<Mutex<KeyringCache>> = OnceLock::new();

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

    fn cache() -> &'static Mutex<KeyringCache> {
        KEYRING_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
    }

    fn cache_key(&self) -> (String, String) {
        (self.service.clone(), self.account.clone())
    }
}

#[cfg(feature = "keyring")]
impl TokenStore for KeyringTokenStore {
    fn get_token(&self) -> Result<Option<String>> {
        let mut cache = Self::cache()
            .lock()
            .map_err(|_| Error::Token("keyring cache poisoned".into()))?;
        if let Some(cached) = cache.get(&self.cache_key()) {
            return cached.clone().map_err(Error::Token);
        }

        let result = match self.entry()?.get_password() {
            Ok(token) => Ok(Some(token)),
            Err(keyring::Error::NoEntry) => Ok(None),
            Err(error) => Err(format!("keyring get: {error}")),
        };
        cache.insert(self.cache_key(), result.clone());
        result.map_err(Error::Token)
    }

    fn set_token(&self, token: &str) -> Result<()> {
        let result = self
            .entry()?
            .set_password(token)
            .map_err(|error| format!("keyring set: {error}"));
        let cached = result.clone().map(|()| Some(token.to_string()));
        Self::cache()
            .lock()
            .map_err(|_| Error::Token("keyring cache poisoned".into()))?
            .insert(self.cache_key(), cached);
        result.map_err(Error::Token)
    }

    fn clear(&self) -> Result<()> {
        let result = match self.entry()?.delete_credential() {
            Ok(()) => Ok(()),
            Err(keyring::Error::NoEntry) => Ok(()),
            Err(error) => Err(format!("keyring clear: {error}")),
        };
        let cached = result.clone().map(|()| None);
        Self::cache()
            .lock()
            .map_err(|_| Error::Token("keyring cache poisoned".into()))?
            .insert(self.cache_key(), cached);
        result.map_err(Error::Token)
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
