# Complex Refactor — nvfp4

**Model:** `mlx-community/Qwen3.5-397B-A17B-nvfp4`

| Metric | Value |
|---|---|
| Thinking tokens | 915 |
| Completion tokens | 1866 |
| Decode tok/s | 34.7 |
| TTFT | 2260ms |
| Total time | 56.0s |

## Thinking

```
Looking at this legacy callback-based code, I need to:

1. Convert to async/await with proper TypeScript types
2. Fix the bug where callback fires before forEach queries complete
3. Preserve exact behavior (including the bug? No - I should fix the bug since it's marked as a bug)
4. Add proper error handling
5. Make it a drop-in replacement

Key issues to address:
- Nested callback hell (pyramid of doom)
- The setTimeout bug - callback fires before order_items queries complete
- No types
- Error handling could be better

For drop-in replacement, I need to maintain the same function signature but can return a Promise. Actually, for true drop-in replacement, I should support both callback and Promise patterns, OR I should convert to async function that returns Promise (which is compatible with callback pattern if the caller handles it).

Actually, for "drop-in replacement" with modern async/await, the best approach is to make it an async function that returns a Promise. Callers using callbacks can wrap it, or I can make it support both patterns.

Let me create a proper TypeScript version with:
1. Proper types for User, Order, OrderItem
2. async/await instead of callbacks
3. Fix the bug where callback fires before all queries complete\n4. Proper error handling with try/catch\n5. Use Promise.all for parallel queries where appropriate

Let me write the refactored code:

```typescript
interface User {\n  id: number;\n  [key: string]: any;\n  orders?: Order[];\n}\n\ninterface Order {\n  id: number;\n  user_id: number;\n  items?: OrderItem[];\n}\n\ninterface OrderItem {\n  id: number;\n  order_id: number;\n  [key: string]: any;\n}\n\nasync function fetchUserData(userId: number): Promise<User | null> {\n  try {\n    // Fetch user\n    const userRows = await db.query('SELECT * FROM users WHERE id = ?', [userId]);\n    \n    if (!userRows || userRows.length === 0) {\n      return null;\n    }\n    \n    const user: User = userRows[0];\n    \n    // Fetch orders\n    const orders = await db.query('SELECT * FROM orders WHERE user_id = ?', [userId]);\n    user.orders = orders || [];\n    \n    // Fetch order items for each order (fixed: wait for all to complete)\n    if (user.orders && user.orders.length > 0) {\n      await Promise.all(\n        user.orders.map(async (order: Order) => {\n          const items = await db.query('SELECT * FROM order_items WHERE order_id = ?', [order.id]);\n          order.items = items || [];\n        })\n      );\n    }\n    \n    return user;\n  } catch (error) {\n    throw error;\n  }\n}\n```\n\nWait, for true drop-in replacement, I should support the callback pattern too. Let me make it support both callback and Promise patterns like many Node.js libraries do.

Actually, re-reading the requirement: "drop-in replacement" - this means existing code that calls this function should continue to work. If existing code passes a callback, I need to support that. But if I'm modernizing to async/await, the calling code would need to change too.\n\nThe most practical approach for a "drop-in replacement" while modernizing is to make the function return a Promise (which is async/await compatible) but also support the callback pattern for backward compatibility. This is what many Node.js libraries did during the callback-to-Promise transition.\n\nLet me create a version that supports both:

```typescript\nfunction fetchUserData(userId: number): Promise<User | null>;\nfunction fetchUserData(userId: number, callback: (err: Error | null, user: User | null) => void): void;\nfunction fetchUserData(\n  userId: number,\n  callback?: (err: Error | null, user: User | null) => void\n): Promise<User | null> | void {\n  // Implementation that supports both patterns\n}\n```\n\nThis way, existing callback code continues to work, and new code can use async/await.

```

## Response



```typescript
// Type definitions
interface OrderItem {\n  id: number;\n  order_id: number;\n  [key: string]: any;\n}\n\ninterface Order {\n  id: number;\n  user_id: number;\n  items?: OrderItem[];\n  [key: string]: any;\n}\n\ninterface User {\n  id: number;\n  [key: string]: any;\n  orders?: Order[];\n}\n\n// Type-safe database query wrapper (assumes db.query returns Promise when no callback)\nasync function dbQuery<T = any>(sql: string, params: any[]): Promise<T[]> {\n  return new Promise((resolve, reject) => {\n    (db as any).query(sql, params, (err: Error | null, rows: T[]) => {\n      if (err) return reject(err);\n      resolve(rows);\n    });\n  });\n}\n\n// Main function with dual callback/Promise support for drop-in replacement\nfunction fetchUserData(userId: number): Promise<User | null>;\nfunction fetchUserData(\n  userId: number,\n  callback: (err: Error | null, user: User | null) => void\n): void;\nfunction fetchUserData(\n  userId: number,\n  callback?: (err: Error | null, user: User | null) => void\n): Promise<User | null> | void {\n  // Internal async implementation\n  const execute = async (): Promise<User | null> => {\n    // Fetch user\n    const userRows = await dbQuery<User>('SELECT * FROM users WHERE id = ?', [userId]);\n\n    if (!userRows || userRows.length === 0) {\n      return null;\n    }\n\n    const user: User = userRows[0];\n\n    // Fetch orders\n    const orders = await dbQuery<Order>('SELECT * FROM orders WHERE user_id = ?', [userId]);\n    user.orders = orders || [];\n\n    // Fetch order items for each order - FIXED: wait for all queries to complete\n    if (user.orders && user.orders.length > 0) {\n      await Promise.all(\n        user.orders.map(async (order: Order) => {\n          const items = await dbQuery<OrderItem>(\n            'SELECT * FROM order_items WHERE order_id = ?',\n            [order.id]\n          );\n          order.items = items || [];\n        })\n      );\n    }\n\n    return user;\n  };\n\n  // Support callback pattern for backward compatibility\n  if (callback) {\n    execute()\n      .then((user) => callback(null, user))\n      .catch((err) => callback(err, null));\n    return;\n  }\n\n  // Return Promise for async/await usage\n  return execute();\n}\n\nexport { fetchUserData, User, Order, OrderItem };\n```

### Key Improvements

| Issue | Original | Refactored |\n|-------|----------|------------|\n| **Callback Hell** | 3+ nested callbacks | Flat async/await structure |\n| **Timing Bug** | `setTimeout` doesn't wait for queries | `Promise.all` ensures all complete |\n| **Types** | None | Full TypeScript interfaces |\n| **Error Handling** | Basic callback errors | Try/catch + Promise rejection |\n| **Compatibility** | Callback only | Callback + Promise (drop-in) |\n| **Maintainability** | Hard to extend | Clean, modular structure |\n\n### Usage Examples

```typescript\n// Legacy callback style (still works)\nfetchUserData(123, (err, user) => {\n  if (err) console.error(err);\n  else console.log(user);\n});\n\n// Modern async/await\nconst user = await fetchUserData(123);\n\n// Promise style\nfetchUserData(123).then(user => console.log(user));\n```

### Notes

- Assumes `db.query` supports callback pattern; wrapped in `dbQuery` for Promise compatibility\n- Returns `null` when user not found (preserves original behavior)\n- All database errors properly propagate to caller\n- `Promise.all` fixes the critical bug where callback fired before order_items queries completed
