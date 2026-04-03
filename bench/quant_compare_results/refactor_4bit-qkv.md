# Complex Refactor — 4bit-qkv

**Model:** `mlx-community/Qwen3.5-397B-A17B-4bit`

| Metric | Value |
|---|---|
| Thinking tokens | 629 |
| Completion tokens | 1408 |
| Decode tok/s | 33.6 |
| TTFT | 2216ms |
| Total time | 44.2s |

## Thinking

```
This is a classic callback hell scenario with several issues:
1. Nested callbacks making code hard to read
2. A bug where the callback fires before async operations complete (the setTimeout is a hack)
3. No type safety
4. No proper error handling

I need to:
1. Convert to async/await
2. Add proper TypeScript types
3. Fix the bug where callback fires before operations complete
4. Maintain the exact same behavior (return structure, error handling)
5. Make it a drop-in replacement (same function signature, but we can modernize it)

Since it needs to be a drop-in replacement, I should keep the callback signature but I can also provide both versions. Actually, re-reading the requirement - it says "modern async/await TypeScript" so I should convert to async/await. For drop-in replacement, I could either:
1. Keep the callback signature but use async/await internally
2. Change to async/await signature

Given it says "modern async/await TypeScript", I'll convert to async/await. For drop-in replacement, I'll note that callers would need to await it, but the behavior is preserved.

Let me write proper TypeScript with:
- Interface types for User, Order, OrderItem
- Proper async/await
- Proper error handling
- Fix the bug where we need to wait for all order items queries to complete

```typescript
interface User {
  id: number;
  [key: string]: any;
}

interface Order {
  id: number;
  user_id: number;
  items?: any[];
  [key: string]: any;
}

interface OrderItem {
  id: number;
  order_id: number;
  [key: string]: any;
}

async function fetchUserData(userId: number): Promise<User | null> {
  // Fetch user
  const [userRows] = await db.query('SELECT * FROM users WHERE id = ?', [userId]);
  
  if (!userRows || (userRows as any[]).length === 0) {
    return null;
  }
  
  const user: User = (userRows as any[])[0];
  
  // Fetch orders
  const [ordersRows] = await db.query('SELECT * FROM orders WHERE user_id = ?', [userId]);
  const orders: Order[] = ordersRows as any[] || [];
  
  // Fetch items for each order (wait for all to complete)
  await Promise.all(
    orders.map(async (order) => {
      const [itemsRows] = await db.query('SELECT * FROM order_items WHERE order_id = ?', [order.id]);
      order.items = itemsRows as any[] || [];
    })
  );
  
  user.orders = orders;
  return user;
}
```

I'm completing the partial thought by finishing the incomplete line and continuing with the next thinking. The code now properly handles the database queries with async/await, ensuring all order items are fetched before returning the user data.

```

## Response



```typescript
// Types
interface User {
  id: number;
  [key: string]: any;
}

interface Order {
  id: number;
  user_id: number;
  items?: any[];
  [key: string]: any;
}

interface OrderItem {
  id: number;
  order_id: number;
  [key: string]: any;
}

// Helper to promisify db.query (adjust based on your DB library)
function query<T = any>(sql: string, params: any[]): Promise<T> {
  return new Promise((resolve, reject) => {
    db.query(sql, params, (err: Error | null, results: any) => {
      if (err) reject(err);
      else resolve(results as T);
    });
  });
}

// Refactored function
export async function fetchUserData(userId: number): Promise<User | null> {
  try {
    // Fetch user
    const userRows: any[] = await query('SELECT * FROM users WHERE id = ?', [userId]);

    if (!userRows || userRows.length === 0) {
      return null;
    }

    const user: User = userRows[0];

    // Fetch orders
    const ordersRows: any[] = await query('SELECT * FROM orders WHERE user_id = ?', [userId]);
    const orders: Order[] = ordersRows || [];

    // Fetch items for each order and wait for all to complete
    await Promise.all(
      orders.map(async (order: Order) => {
        const itemsRows: any[] = await query(
          'SELECT * FROM order_items WHERE order_id = ?',
          [order.id]
        );
        order.items = itemsRows || [];
      })
    );

    // Attach orders to user
    user.orders = orders;

    return user;
  } catch (error) {
    // Re-throw to allow caller to handle
    throw error;
  }
}
```

### Key Improvements

| Issue | Original | Refactored |
|-------|----------|------------|
| **Callback Hell** | Deeply nested callbacks | Flat async/await flow |
| **Race Condition** | `setTimeout` hack | `Promise.all` ensures completion |
| **Type Safety** | Dynamic `var` | TypeScript interfaces |
| **Error Handling** | Manual `if (err)` checks | `try/catch` with re-throw |
| **Maintainability** | Hard to follow | Clear sequential logic |

### Usage Example

```typescript
// Before
fetchUserData(123, (err, user) => {
  if (err) throw err;
  console.log(user);
});

// After
try {
  const user = await fetchUserData(123);
  console.log(user);
} catch (error) {
  console.error(error);
}
```

### Migration Note

To maintain a drop-in replacement with the original callback signature, you can wrap the async function:

```typescript
export function fetchUserData(
  userId: number,
  callback: (err: Error | null, user: User | null) => void
): void {
  fetchUserDataAsync(userId)
    .then(user => callback(null, user))
    .catch(err => callback(err, null));
}
```

Replace `fetchUserDataAsync` with the async function defined above.
